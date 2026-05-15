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

  return { follow, stopMoving, goTo, goNear, mine, mineNearest, chat, getStatus };
}

module.exports = { setupActions };
