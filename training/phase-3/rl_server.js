/**
 * rl_server.js
 * ============
 * Mineflayer bot that exposes an HTTP API for the Python RL training loop.
 * 
 * Endpoints:
 *   GET  /state        → current game state as JSON
 *   POST /action       → send an action, returns result + reward signals
 *   POST /reset        → respawn bot, reset episode
 *   GET  /health       → health check
 * 
 * Usage:
 *   npm install mineflayer mineflayer-pathfinder express
 *   node bot/rl_server.js --host localhost --port 25565 --api-port 3001
 */

const mineflayer   = require('mineflayer');
const pathfinder   = require('mineflayer-pathfinder').pathfinder;
const { Movements, goals } = require('mineflayer-pathfinder');
const express      = require('express');
const { Vec3 }     = require('vec3');

// ── Config ──────────────────────────────────────────────────────────────────
const MC_HOST    = process.env.MC_HOST    || 'localhost';
const MC_PORT    = parseInt(process.env.MC_PORT    || '25565');
const API_PORT   = parseInt(process.env.API_PORT   || '3001');
const BOT_USER   = process.env.BOT_USER   || 'MineAgent_RL';

// ── State ────────────────────────────────────────────────────────────────────
let bot = null;
let lastResult = {};
let episodeStartHp = 20;
let stepCount = 0;

// ── Bot Factory ───────────────────────────────────────────────────────────────
function createBot() {
  bot = mineflayer.createBot({
    host:     MC_HOST,
    port:     MC_PORT,
    username: BOT_USER,
    version:  '1.20.1',
  });

  bot.loadPlugin(pathfinder);

  bot.once('spawn', () => {
    const defaultMove = new Movements(bot);
    defaultMove.allowSprinting = true;
    bot.pathfinder.setMovements(defaultMove);
    episodeStartHp = bot.health;
    console.log(`[RL Server] Bot spawned. HP: ${bot.health} Food: ${bot.food}`);
  });

  bot.on('death', () => {
    lastResult.died = true;
    console.log('[RL Server] Bot died!');
  });

  bot.on('error', err => console.error('[RL Server] Bot error:', err.message));
  bot.on('end', () => {
    console.log('[RL Server] Bot disconnected, reconnecting in 3s...');
    setTimeout(createBot, 3000);
  });
}

// ── Get Game State ────────────────────────────────────────────────────────────
function getState() {
  if (!bot || !bot.entity) {
    return { hp: 0, food: 0, pos: { x: 0, y: 64, z: 0 }, inv: {}, nearby: [], mobs: [], time: 0, raining: false };
  }

  // Inventory
  const inv = {};
  bot.inventory.items().forEach(item => {
    if (item) inv[item.name] = (inv[item.name] || 0) + item.count;
  });

  // Nearby blocks (within 8 blocks)
  const nearby = [];
  const pos = bot.entity.position;
  const interestingBlocks = [
    'oak_log', 'birch_log', 'spruce_log', 'cobblestone', 'stone',
    'coal_ore', 'iron_ore', 'gold_ore', 'diamond_ore',
    'crafting_table', 'furnace', 'chest', 'dirt', 'sand',
  ];

  for (const blockName of interestingBlocks) {
    try {
      const blockId = bot.registry.blocksByName[blockName]?.id;
      if (!blockId) continue;
      const block = bot.findBlock({ matching: blockId, maxDistance: 8 });
      if (block) {
        const dist = bot.entity.position.distanceTo(block.position);
        nearby.push({ name: blockName, dist: Math.round(dist * 10) / 10 });
      }
    } catch (e) { /* block not found */ }
  }

  // Nearby entities (mobs)
  const mobs = [];
  const hostileMobs = ['zombie', 'skeleton', 'creeper', 'spider', 'witch', 'enderman'];
  Object.values(bot.entities).forEach(entity => {
    if (!entity || entity === bot.entity) return;
    const dist = entity.position.distanceTo(bot.entity.position);
    if (dist > 16) return;
    const isHostile = hostileMobs.includes(entity.name);
    mobs.push({
      name:    entity.name || 'unknown',
      dist:    Math.round(dist * 10) / 10,
      hp:      Math.round((entity.health || 20)),
      hostile: isHostile,
    });
  });

  return {
    hp:      Math.round(bot.health || 0),
    food:    Math.round(bot.food   || 0),
    pos:     {
      x: Math.round(pos.x),
      y: Math.round(pos.y),
      z: Math.round(pos.z),
    },
    inv,
    nearby:  nearby.sort((a, b) => a.dist - b.dist).slice(0, 8),
    mobs:    mobs.sort((a, b)    => a.dist - b.dist).slice(0, 6),
    time:    bot.time?.timeOfDay || 0,
    raining: bot.isRaining || false,
  };
}

// ── Action Executor ───────────────────────────────────────────────────────────
async function executeAction(actionName, params = {}) {
  if (!bot || !bot.entity) return { invalid: true };
  
  const prevHp  = bot.health;
  const prevInv = {};
  bot.inventory.items().forEach(i => { if (i) prevInv[i.name] = (prevInv[i.name] || 0) + i.count; });

  const result = { action: actionName, collected: {}, crafted: false, smelted: false,
                   killed_mob: false, ate: false, died: false, invalid: false,
                   damage_taken: 0, goal_achieved: false };

  try {
    switch (actionName) {
      case 'MOVE_FORWARD':
        bot.setControlState('forward', true);
        await sleep(400);
        bot.setControlState('forward', false);
        break;

      case 'MOVE_BACK':
        bot.setControlState('back', true);
        await sleep(400);
        bot.setControlState('back', false);
        break;

      case 'MOVE_LEFT':
        bot.setControlState('left', true);
        await sleep(400);
        bot.setControlState('left', false);
        break;

      case 'MOVE_RIGHT':
        bot.setControlState('right', true);
        await sleep(400);
        bot.setControlState('right', false);
        break;

      case 'JUMP':
        bot.setControlState('jump', true);
        await sleep(200);
        bot.setControlState('jump', false);
        break;

      case 'MINE': {
        const mineTargets = ['oak_log', 'birch_log', 'cobblestone', 'stone',
                              'coal_ore', 'iron_ore', 'gold_ore', 'diamond_ore', 'dirt', 'sand'];
        let mined = false;
        for (const blockName of mineTargets) {
          const blockId = bot.registry.blocksByName[blockName]?.id;
          if (!blockId) continue;
          const block = bot.findBlock({ matching: blockId, maxDistance: 5 });
          if (block) {
            await bot.dig(block);
            result.collected[blockName] = 1;
            mined = true;
            break;
          }
        }
        if (!mined) result.invalid = true;
        break;
      }

      case 'ATTACK': {
        // Attack nearest hostile mob
        const hostile = Object.values(bot.entities).find(e =>
          e && e !== bot.entity &&
          ['zombie', 'skeleton', 'creeper', 'spider'].includes(e.name) &&
          e.position.distanceTo(bot.entity.position) < 5
        );
        if (hostile) {
          await bot.attack(hostile);
          if (!hostile.isValid) result.killed_mob = true;
        } else {
          result.invalid = true;
        }
        break;
      }

      case 'CRAFT': {
        // Try to craft any item we have ingredients for
        const recipes_priority = ['iron_pickaxe', 'stone_pickaxe', 'wooden_pickaxe',
                                   'iron_sword', 'crafting_table', 'stick', 'wooden_planks'];
        let crafted = false;
        for (const itemName of recipes_priority) {
          const item = bot.registry.itemsByName[itemName];
          if (!item) continue;
          const recipes = bot.recipesFor(item.id, null, 1, null);
          if (recipes.length > 0) {
            await bot.craft(recipes[0], 1, null);
            result.crafted = true;
            crafted = true;
            break;
          }
        }
        if (!crafted) result.invalid = true;
        break;
      }

      case 'EAT': {
        const foods = ['bread', 'cooked_beef', 'cooked_chicken', 'apple', 'carrot'];
        let ate = false;
        for (const foodName of foods) {
          const foodItem = bot.inventory.findInventoryItem(
            bot.registry.itemsByName[foodName]?.id);
          if (foodItem) {
            await bot.equip(foodItem, 'hand');
            await bot.consume();
            result.ate = true;
            ate = true;
            break;
          }
        }
        if (!ate) result.invalid = true;
        break;
      }

      case 'PLACE': {
        const placeItems = ['torch', 'crafting_table', 'furnace', 'chest'];
        let placed = false;
        for (const itemName of placeItems) {
          const item = bot.inventory.findInventoryItem(
            bot.registry.itemsByName[itemName]?.id);
          if (item) {
            await bot.equip(item, 'hand');
            const refBlock = bot.blockAt(bot.entity.position.offset(0, -1, 0));
            if (refBlock) {
              await bot.placeBlock(refBlock, new Vec3(0, 1, 0));
              placed = true;
              break;
            }
          }
        }
        if (!placed) result.invalid = true;
        break;
      }

      case 'WAIT':
        await sleep(500);
        break;

      default:
        result.invalid = true;
    }
  } catch (err) {
    console.warn(`[RL Server] Action ${actionName} failed:`, err.message);
    result.invalid = true;
  }

  // Check damage taken
  result.damage_taken = Math.max(0, prevHp - (bot.health || 0));
  result.died = lastResult.died || false;
  lastResult.died = false;

  // Check new items collected
  const newInv = {};
  bot.inventory.items().forEach(i => { if (i) newInv[i.name] = (newInv[i.name] || 0) + i.count; });
  for (const [item, count] of Object.entries(newInv)) {
    const gained = count - (prevInv[item] || 0);
    if (gained > 0) result.collected[item] = gained;
  }

  return result;
}

// ── HTTP API ──────────────────────────────────────────────────────────────────
const app = express();
app.use(express.json());

app.get('/health', (req, res) => {
  res.json({ status: 'ok', connected: bot?.entity != null, step: stepCount });
});

app.get('/state', (req, res) => {
  res.json(getState());
});

app.post('/action', async (req, res) => {
  const { action } = req.body;
  if (!action) return res.status(400).json({ error: 'action required' });
  stepCount++;
  const result = await executeAction(action, req.body);
  res.json(result);
});

app.post('/reset', async (req, res) => {
  stepCount = 0;
  lastResult = {};
  // In a real setup, you'd teleport the bot to spawn or run /kill
  // For now just wait for natural respawn
  if (bot) {
    try {
      bot.chat('/kill');   // requires op or cheats enabled
      await sleep(2000);
    } catch (e) { /* ignore */ }
  }
  res.json({ status: 'reset', goal: req.body.goal });
});

// ── Start ─────────────────────────────────────────────────────────────────────
function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }

app.listen(API_PORT, () => {
  console.log(`[RL Server] HTTP API listening on port ${API_PORT}`);
  console.log(`[RL Server] Connecting to Minecraft ${MC_HOST}:${MC_PORT}...`);
  createBot();
});
