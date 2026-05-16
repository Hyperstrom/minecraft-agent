/**
 * actions.js
 * Wraps Mineflayer APIs into named, callable action functions.
 * The LLM (Phase 2) will call these by name via the Tool Registry.
 */

const { pathfinder, Movements } = require('mineflayer-pathfinder');
const { GoalFollow, GoalBlock, GoalNear } = require('mineflayer-pathfinder').goals;

function setupActions(bot) {

  /** Navigate to and follow an entity continuously. */
  function follow(target, distance = 2) {
    if (!target) { bot.chat('No target to follow.'); return; }
    const movements = new Movements(bot);
    bot.pathfinder.setMovements(movements);
    bot.pathfinder.setGoal(new GoalFollow(target, distance), true);
  }

  /** Stop all pathfinder movement. */
  function stopMoving() {
    bot.pathfinder.stop();
    bot.clearControlStates();
  }

  /** Walk to absolute coordinates. */
  function goTo(x, y, z) {
    const movements = new Movements(bot);
    bot.pathfinder.setMovements(movements);
    bot.pathfinder.setGoal(new GoalBlock(x, y, z));
  }

  /** Approach within `distance` blocks of coordinates. */
  function goNear(x, y, z, distance = 3) {
    const movements = new Movements(bot);
    bot.pathfinder.setMovements(movements);
    bot.pathfinder.setGoal(new GoalNear(x, y, z, distance));
  }

  /** Dig a block reference (obtained via bot.blockAt). */
  async function mine(block) {
    if (!block) return false;
    try {
      await bot.dig(block);
      return true;
    } catch (err) {
      console.error(`[actions] mine error: ${err.message}`);
      return false;
    }
  }

  /** Find the nearest block by name and dig it. */
  async function mineNearest(blockName) {
    const block = bot.findBlock({
      matching: bot.registry.blocksByName[blockName]?.id,
      maxDistance: 32,
    });
    if (!block) { bot.chat(`No ${blockName} nearby.`); return false; }
    await goNear(block.position.x, block.position.y, block.position.z, 3);
    return mine(block);
  }

  /** Send a chat message. */
  function chat(message) {
    bot.chat(message);
  }

  /** Return a compact status object (used by commands). */
  function getStatus() {
    return {
      health:   bot.health,
      food:     bot.food,
      position: bot.entity?.position ?? null,
    };
  }

  /**
   * SEEK — find nearest block within 256 blocks and walk to it.
   * Used when the LLM decides target is not in immediate nearby_blocks.
   */
  async function seekBlock(blockName) {
    // Map friendly names to Minecraft block IDs
    const nameMap = {
      'food':        null,  // special case handled below
      'oak_log':     'oak_log',
      'birch_log':   'birch_log',
      'acacia_log':  'acacia_log',
      'jungle_log':  'jungle_log',
      'spruce_log':  'spruce_log',
      'dark_oak_log':'dark_oak_log',
      'coal_ore':    'coal_ore',
      'iron_ore':    'iron_ore',
      'stone':       'stone',
      'cobblestone': 'cobblestone',
      'water':       'water',
      'grass_block': 'grass_block',
    };

    const target = nameMap[blockName] || blockName;

    const blockDef = bot.registry.blocksByName[target];
    if (!blockDef) {
      bot.chat(`Don't know how to seek: ${blockName}`);
      return false;
    }

    const block = bot.findBlock({
      matching:    blockDef.id,
      maxDistance: 256,
    });

    if (!block) {
      bot.chat(`Could not find ${blockName} within 256 blocks — exploring further`);
      // Explore a random direction to find more blocks
      const dirs = [[50,0,0],[-50,0,0],[0,0,50],[0,0,-50]];
      const d = dirs[Math.floor(Math.random() * dirs.length)];
      const pos = bot.entity.position;
      goTo(Math.floor(pos.x + d[0]), Math.floor(pos.y), Math.floor(pos.z + d[2]));
      return false;
    }

    bot.chat(`Seeking ${blockName} at (${Math.floor(block.position.x)}, ${Math.floor(block.position.y)}, ${Math.floor(block.position.z)})`);
    await goNear(block.position.x, block.position.y, block.position.z, 4);
    return true;
  }

  /** Move bot to within 5 blocks of a player by username. */
  function goNearPlayer(username) {
    const player = bot.players[username]?.entity;
    if (!player) return false;
    const p = player.position;
    goNear(p.x, p.y, p.z, 5);
    return true;
  }

  return { follow, stopMoving, goTo, goNear, mine, mineNearest, seekBlock, goNearPlayer, chat, getStatus };
}

module.exports = { setupActions };
