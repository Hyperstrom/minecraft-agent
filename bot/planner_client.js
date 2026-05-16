/**
 * planner_client.js  —  Autonomous planning loop for Phase 2.
 *
 * Every PLAN_INTERVAL ms:
 *   1. Extract Observation JSON from live bot state
 *   2. Enrich state with goal progress + recent action history
 *   3. POST to /plan endpoint
 *   4. Dispatch returned action to the action executor
 *   5. Check if goal is now complete → auto-stop
 */

const axios = require('axios');
const config = require('./config');
const log    = require('./logger');
const { extractState, getInventory } = require('./state_extractor');
const { parseGoal, checkProgress, getProgressText } = require('./goal_tracker');

const PLAN_INTERVAL      = 5000;   // ms between planning ticks
const PLAN_TIMEOUT       = 60000;  // ms — 60s handles Llama cold start
const MAX_RECENT_ACTIONS = 5;      // how many past actions to include in prompt

let _timer         = null;
let _busy          = false;
let _goal          = null;
let _parsedGoal    = null;    // structured {type, items, count} from goal_tracker
let _enabled       = false;
let _bot           = null;
let _actions       = null;
let _recentActions = [];      // last N action strings for LLM context

// ── Public API ────────────────────────────────────────────────────

function start(bot, actions) {
  if (_timer) stop();
  _bot           = bot;
  _actions       = actions;
  _enabled       = true;
  _recentActions = [];

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

  try { if (_actions) _actions.stopMoving(); } catch (_) {}
  log.info('[Planner] Stopped.');
}

function setGoal(goal) {
  _goal       = goal;
  _parsedGoal = parseGoal(goal);
  _recentActions = [];   // clear history on new goal

  log.info(`[Planner] Goal set: "${goal}" → parsed: ${JSON.stringify(_parsedGoal)}`);

  axios.post(`${config.backendUrl}/goal`, { goal }, {
    timeout: 3000,
    headers: { 'Content-Type': 'application/json' },
  }).catch(err => log.error(`[Planner] Failed to sync goal to backend: ${err.message}`));
}

function getGoal() { return _goal; }

// ── Internal ──────────────────────────────────────────────────────

async function _tick(bot, actions) {
  const state    = extractState(bot, _goal);
  const inventory = getInventory(bot);

  // ── Goal completion check ────────────────────────────────────
  if (_parsedGoal) {
    const { done, progressText } = checkProgress(inventory, _parsedGoal);
    if (done) {
      log.info(`[Planner] GOAL COMPLETE: ${progressText}`);
      bot.chat(`Goal complete! ${progressText}`);
      stop();
      return;
    }
    // Attach progress to state so backend can use it in prompt
    state.goal_progress = progressText;
    log.info(`[Planner] Progress: ${progressText}`);
  }

  // ── Attach recent action history ─────────────────────────────
  state.recent_actions = _recentActions.slice(-MAX_RECENT_ACTIONS);

  // ── Request plan from backend ────────────────────────────────
  let resp;
  try {
    resp = await axios.post(`${config.backendUrl}/plan`, state, {
      timeout: PLAN_TIMEOUT,
      headers: { 'Content-Type': 'application/json' },
    });
  } catch (err) {
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

  // Record this action for context
  _recentActions.push(`${action}(${JSON.stringify(params)})`);
  if (_recentActions.length > MAX_RECENT_ACTIONS) _recentActions.shift();

  await _dispatch(bot, actions, action, params);
}

async function _dispatch(bot, actions, action, params = {}) {
  switch (action) {

    case 'SEEK':
      await actions.seekBlock(params.target || params.block || 'oak_log');
      break;

    case 'MOVE': {
      const dir  = (params.direction || 'north').toLowerCase();
      const dist = params.distance || 20;
      const offsets = { north:[0,0,-1], south:[0,0,1], east:[1,0,0], west:[-1,0,0] };
      const off  = offsets[dir] || [0, 0, 0];
      const pos  = bot.entity.position;
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

    case 'CRAFT':
      await actions.craftItem(params.item, params.count || 1);
      break;

    case 'COLLECT':
      // Bot auto-collects on proximity; no extra action needed
      log.info(`[Planner] COLLECT ${params.item} — bot collects on proximity`);
      break;

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
