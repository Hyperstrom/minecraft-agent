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

  /**
   * Craft an item by name. Automatically:
   *  - Tries 2×2 inventory crafting first (no table needed)
   *  - If 3×3 recipe, finds/pathfinds to a crafting table
   *  - Falls back to placing a crafting table if one is in inventory
   */
  async function craftItem(itemName, count = 1) {
    const itemDef = bot.registry.itemsByName[itemName];
    if (!itemDef) {
      bot.chat(`Don't know item: ${itemName}`);
      return false;
    }

    // ── Try 2×2 crafting (no table) ─────────────────────────────
    const recipesNoTable = bot.recipesFor(itemDef.id, null, count, null);
    if (recipesNoTable.length > 0) {
      try {
        await bot.craft(recipesNoTable[0], count, null);
        bot.chat(`Crafted ${count}x ${itemName}`);
        return true;
      } catch (e) {
        // Probably needs 3×3 — fall through
      }
    }

    // ── Try 3×3 crafting (requires crafting table) ───────────────
    const recipesWithTable = bot.recipesFor(itemDef.id, null, count, true);
    if (recipesWithTable.length === 0) {
      // Try with undefined (all recipes)
      const allRecipes = bot.recipesFor(itemDef.id, null, count, null);
      if (allRecipes.length === 0) {
        bot.chat(`No recipe found for ${itemName} — missing ingredients?`);
        return false;
      }
    }

    // Find a nearby crafting table
    const tableId = bot.registry.blocksByName['crafting_table']?.id;
    let table = tableId ? bot.findBlock({ matching: tableId, maxDistance: 32 }) : null;

    // If no table nearby, try to place one from inventory
    if (!table) {
      const tableItem = bot.inventory.items().find(i => i.name === 'crafting_table');
      if (tableItem) {
        try {
          const ground = bot.blockAt(bot.entity.position.offset(0, -1, 0));
          const refBlock = bot.blockAt(bot.entity.position.offset(1, 0, 0));
          await bot.equip(tableItem, 'hand');
          await bot.placeBlock(ground, new (require('vec3'))(0, 1, 0));
          table = bot.findBlock({ matching: tableId, maxDistance: 5 });
          bot.chat('Placed crafting table');
        } catch (e) {
          bot.chat('Could not place crafting table');
        }
      }
    }

    if (!table) {
      bot.chat(`Need a crafting table to craft ${itemName}. Seeking one...`);
      if (tableId) {
        const farTable = bot.findBlock({ matching: tableId, maxDistance: 128 });
        if (farTable) {
          await goNear(farTable.position.x, farTable.position.y, farTable.position.z, 3, 30000);
          table = bot.blockAt(farTable.position);
        }
      }
      if (!table) {
        bot.chat(`Could not find a crafting table anywhere nearby`);
        return false;
      }
    }

    // Navigate to the table
    await goNear(table.position.x, table.position.y, table.position.z, 3, 20000);

    // Craft at table
    const recipes = bot.recipesFor(itemDef.id, null, count, table);
    if (recipes.length === 0) {
      bot.chat(`Missing ingredients for ${itemName}`);
      return false;
    }
    try {
      await bot.craft(recipes[0], count, table);
      bot.chat(`Crafted ${count}x ${itemName} at crafting table!`);
      return true;
    } catch (e) {
      bot.chat(`Craft failed: ${e.message}`);
      return false;
    }
  }

  return { follow, stopMoving, goTo, goNear, mine, mineNearest, seekBlock, goNearPlayer, craftItem, chat, getStatus };
}

module.exports = { setupActions };
