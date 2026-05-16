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

  /**
   * Approach within `distance` blocks of coordinates.
   * Returns a Promise that resolves when the bot arrives (or times out).
   */
  function goNear(x, y, z, distance = 3, timeoutMs = 30000) {
    return new Promise((resolve) => {
      const movements = new Movements(bot);
      bot.pathfinder.setMovements(movements);
      bot.pathfinder.setGoal(new GoalNear(x, y, z, distance));

      let resolved = false;
      const done = () => {
        if (resolved) return;
        resolved = true;
        clearTimeout(timer);
        // Remove BOTH listeners to prevent memory leak
        bot.removeListener('goal_reached', done);
        bot.removeListener('path_stop',    done);
        resolve();
      };
      const timer = setTimeout(done, timeoutMs);

      bot.once('goal_reached', done);
      bot.once('path_stop',    done);
    });
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
   * SEEK — find nearest block within 256 blocks, walk to it, then mine it.
   * This completes the full seek-and-collect cycle in one action.
   */
  async function seekBlock(blockName) {
    const nameMap = {
      'oak_log':      'oak_log',
      'birch_log':    'birch_log',
      'acacia_log':   'acacia_log',
      'jungle_log':   'jungle_log',
      'spruce_log':   'spruce_log',
      'dark_oak_log': 'dark_oak_log',
      'coal_ore':     'coal_ore',
      'iron_ore':     'iron_ore',
      'stone':        'stone',
      'cobblestone':  'cobblestone',
    };

    const target = nameMap[blockName] || blockName;
    const blockDef = bot.registry.blocksByName[target];

    if (!blockDef) {
      bot.chat(`Don't know block: ${blockName}`);
      return false;
    }

    const block = bot.findBlock({ matching: blockDef.id, maxDistance: 256 });

    if (!block) {
      // No block found — explore randomly to find more terrain
      const dirs = [[60,0,0],[-60,0,0],[0,0,60],[0,0,-60]];
      const d = dirs[Math.floor(Math.random() * dirs.length)];
      const pos = bot.entity.position;
      bot.chat(`No ${blockName} found, exploring...`);
      await goNear(
        Math.floor(pos.x + d[0]), Math.floor(pos.y), Math.floor(pos.z + d[2]),
        3, 20000
      );
      return false;
    }

    const p = block.position;
    bot.chat(`Going to ${blockName} at (${Math.floor(p.x)},${Math.floor(p.y)},${Math.floor(p.z)})`);

    // Navigate to the block (waits until arrived or 30s timeout)
    await goNear(p.x, p.y, p.z, 3, 30000);

    // Now mine it
    const fresh = bot.blockAt(block.position);
    if (fresh && fresh.name === target) {
      try {
        await bot.dig(fresh);
        bot.chat(`Collected ${blockName}!`);
        return true;
      } catch (e) {
        // Pathfinder got close enough, try mineNearest as fallback
        return mineNearest(target);
      }
    }

    // Block might have moved (chunk update) — try mineNearest
    return mineNearest(target);
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
