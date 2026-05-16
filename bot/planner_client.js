/**
 * planner_client.js  —  Autonomous planning loop for Phase 2.
 *
 * Every PLAN_INTERVAL ms:
 *   1. Extract Observation JSON from live bot state
 *   2. POST to /plan endpoint
 *   3. Dispatch returned action to the action executor
 */

const axios = require('axios');
const config = require('./config');
const log    = require('./logger');
const { extractState } = require('./state_extractor');

const PLAN_INTERVAL = 5000;   // ms between planning ticks
const PLAN_TIMEOUT  = 60000;  // ms — 60s handles Llama cold start (first load ~60s)

let _timer    = null;
let _busy     = false;
let _goal     = null;
let _enabled  = false;
let _bot      = null;   // stored so stop() can cancel movement
let _actions  = null;

// ── Public API ────────────────────────────────────────────────────

function start(bot, actions) {
  if (_timer) stop();
  _bot     = bot;
  _actions = actions;
  _enabled = true;
  log.info(`[Planner] Started — interval ${PLAN_INTERVAL / 1000}s, backend ${config.backendUrl}`);
  log.info('[Planner] Waiting 10s for backend warmup before first plan...');

  setTimeout(() => {
    if (!_enabled) return;
    _timer = setInterval(async () => {
      if (!_enabled || _busy) return;
      _busy = true;
      try {
        await _tick(bot, actions);
      } catch (err) {
        log.error(`[Planner] Cycle error: ${err.message}`);
      } finally {
        _busy = false;
      }
    }, PLAN_INTERVAL);
    log.info('[Planner] First plan tick starting now!');
  }, 10000);
}

function stop() {
  _enabled = false;
  if (_timer) { clearInterval(_timer); _timer = null; }

  // Cancel any active pathfinding immediately
  try {
    if (_actions) _actions.stopMoving();
  } catch (_) {}

  log.info('[Planner] Stopped.');
}

function setGoal(goal) {
  _goal = goal;
  log.info(`[Planner] Goal set: "${goal}"`);

  axios.post(`${config.backendUrl}/goal`, { goal }, {
    timeout: 3000,
    headers: { 'Content-Type': 'application/json' },
  }).catch(err => log.error(`[Planner] Failed to sync goal to backend: ${err.message}`));
}


function getGoal() { return _goal; }

// ── Internal ──────────────────────────────────────────────────────

async function _tick(bot, actions) {
  const state = extractState(bot, _goal);

  let resp;
  try {
    resp = await axios.post(`${config.backendUrl}/plan`, state, {
      timeout: PLAN_TIMEOUT,
      headers: { 'Content-Type': 'application/json' },
    });
  } catch (err) {
    // Transient connection errors — retry once after 2s
    if (err.code === 'ECONNRESET' || err.code === 'ECONNREFUSED' || err.code === 'ETIMEDOUT') {
      log.warn(`[Planner] /plan ${err.code} — retrying in 2s...`);
      await new Promise(r => setTimeout(r, 2000));
      try {
        resp = await axios.post(`${config.backendUrl}/plan`, state, {
          timeout: PLAN_TIMEOUT,
          headers: { 'Content-Type': 'application/json' },
        });
      } catch (err2) {
        log.error(`[Planner] /plan retry failed: ${err2.message}`);
        return;
      }
    } else {
      log.error(`[Planner] /plan request failed: ${err.message}`);
      return;
    }

  }

  const { action, params, reasoning, source } = resp.data;
  log.info(`[Planner] [${source}] ${action}(${JSON.stringify(params)}) — ${reasoning}`);

  await _dispatch(bot, actions, action, params);
}

async function _dispatch(bot, actions, action, params = {}) {
  switch (action) {

    case 'SEEK':
      await actions.seekBlock(params.target || params.block || 'oak_log');
      break;

    case 'MOVE': {
      const dir = (params.direction || 'north').toLowerCase();
      const dist = params.distance || 5;
      const offsets = { north:[0,0,-1], south:[0,0,1], east:[1,0,0], west:[-1,0,0] };
      const off = offsets[dir] || [0, 0, 0];
      const pos = bot.entity.position;
      actions.goTo(
        Math.floor(pos.x + off[0] * dist),
        Math.floor(pos.y),
        Math.floor(pos.z + off[2] * dist),
      );
      break;
    }

    case 'MINE':
      await actions.mineNearest(params.block || 'stone');
      break;

    case 'COLLECT':
      log.info(`[Planner] COLLECT ${params.item} — not yet automated, bot will pick up on proximity`);
      break;

    case 'CRAFT': {
      try {
        const itemDef = bot.registry.itemsByName[params.item];
        if (!itemDef) { log.warn(`[Planner] Unknown item: ${params.item}`); break; }
        const recipe = bot.recipesFor(itemDef.id, null, 1, null)[0];
        if (recipe) await bot.craft(recipe, params.count || 1, null);
        else log.warn(`[Planner] No recipe for: ${params.item}`);
      } catch (e) { log.error(`[Planner] CRAFT error: ${e.message}`); }
      break;
    }

    case 'EAT': {
      try {
        const itemDef = bot.registry.itemsByName[params.item];
        if (!itemDef) { log.warn(`[Planner] Unknown food: ${params.item}`); break; }
        const food = bot.inventory.findInventoryItem(itemDef.id, null);
        if (food) { await bot.equip(food, 'hand'); await bot.consume(); }
        else log.warn(`[Planner] No ${params.item} in inventory`);
      } catch (e) { log.error(`[Planner] EAT error: ${e.message}`); }
      break;
    }

    case 'CHAT':
      actions.chat(params.message || '...');
      break;

    case 'FOLLOW': {
      const p = Object.values(bot.players).find(pl => pl.username === params.player)?.entity;
      if (p) actions.follow(p);
      else log.warn(`[Planner] Player not found: ${params.player}`);
      break;
    }

    case 'GOTO':
      if (params.x !== undefined) actions.goTo(params.x, params.y, params.z);
      break;

    case 'STOP':
      actions.stopMoving();
      break;

    case 'IDLE':
      break; // intentional no-op

    default:
      log.warn(`[Planner] Unknown action: ${action}`);
  }
}

module.exports = { start, stop, setGoal, getGoal };
