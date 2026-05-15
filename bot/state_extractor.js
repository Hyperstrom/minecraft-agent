/**
 * state_extractor.js
 * Converts live Mineflayer bot data into a structured Observation JSON.
 */

const INTERESTING_BLOCKS = new Set([
  'oak_log', 'birch_log', 'spruce_log', 'jungle_log', 'dark_oak_log',
  'coal_ore', 'iron_ore', 'gold_ore', 'diamond_ore', 'emerald_ore',
  'crafting_table', 'furnace', 'chest', 'water', 'lava',
  'grass_block', 'dirt', 'stone', 'sand', 'gravel',
]);

/**
 * Build the full Observation JSON from a live bot instance.
 * @param {object} bot  - Mineflayer bot
 * @param {string} goal - Current high-level goal (optional)
 */
function extractState(bot, goal = null) {
  return {
    player: {
      username:   bot.username,
      position:   roundPosition(bot.entity?.position ?? null),
      health:     bot.health     ?? null,
      food:       bot.food       ?? null,
      saturation: bot.foodSaturation ?? null,
    },
    inventory:       getInventory(bot),
    nearby_entities: getNearbyEntities(bot),
    nearby_blocks:   getNearbyBlocks(bot),
    environment: {
      time_of_day: getTimeOfDay(bot),
      weather:     bot.isRaining ? 'rain' : 'clear',
      dimension:   'overworld',           // extended later
    },
    goal:      goal,
    timestamp: new Date().toISOString(),
  };
}

/** Round a Vec3 position to 1 decimal place. */
function roundPosition(pos) {
  if (!pos) return null;
  return {
    x: Math.round(pos.x * 10) / 10,
    y: Math.round(pos.y * 10) / 10,
    z: Math.round(pos.z * 10) / 10,
  };
}

/** Aggregate inventory items by name. */
function getInventory(bot) {
  const inv = {};
  for (const item of bot.inventory.items()) {
    inv[item.name] = (inv[item.name] ?? 0) + item.count;
  }
  return inv;
}

/** Return up to 10 nearest entities within radius blocks. */
function getNearbyEntities(bot, radius = 16) {
  if (!bot.entity) return [];
  const result = [];

  for (const entity of Object.values(bot.entities)) {
    if (entity === bot.entity || !entity.position) continue;
    const dist = bot.entity.position.distanceTo(entity.position);
    if (dist > radius) continue;

    result.push({
      type:     entity.type,
      name:     entity.name || entity.username || entity.type,
      distance: Math.round(dist * 10) / 10,
      position: roundPosition(entity.position),
    });
  }

  return result.sort((a, b) => a.distance - b.distance).slice(0, 10);
}

/** Return up to 15 nearby "interesting" blocks within radius. */
function getNearbyBlocks(bot, radius = 5) {
  if (!bot.entity) return [];
  const base = bot.entity.position.floored();
  const result = [];

  for (let x = -radius; x <= radius; x++) {
    for (let y = -radius; y <= radius; y++) {
      for (let z = -radius; z <= radius; z++) {
        const block = bot.blockAt(base.offset(x, y, z));
        if (!block || !INTERESTING_BLOCKS.has(block.name)) continue;
        const dist = Math.sqrt(x * x + y * y + z * z);
        result.push({
          name:     block.name,
          distance: Math.round(dist * 10) / 10,
          position: { x: base.x + x, y: base.y + y, z: base.z + z },
        });
      }
    }
  }

  return result.sort((a, b) => a.distance - b.distance).slice(0, 15);
}

/** Map Minecraft time-of-day ticks to a human label. */
function getTimeOfDay(bot) {
  const t = bot.time?.timeOfDay ?? 0;
  if (t < 6000)  return 'morning';
  if (t < 12000) return 'noon';
  if (t < 13000) return 'sunset';
  return 'night';
}

module.exports = {
  extractState,
  getInventory,
  getNearbyEntities,
  getNearbyBlocks,
  getTimeOfDay,
  roundPosition,
};
