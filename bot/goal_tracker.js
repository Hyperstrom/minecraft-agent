/**
 * goal_tracker.js — Parse goal text and detect completion from inventory.
 *
 * Handles goals like:
 *   "collect 10 wood blocks"  → {item: 'oak_log', count: 10, type: 'collect'}
 *   "gather 5 oak logs"       → {item: 'oak_log', count: 5,  type: 'collect'}
 *   "mine 3 stone"            → {item: 'cobblestone', count: 3, type: 'collect'}
 *   "craft a wooden pickaxe"  → {item: 'wooden_pickaxe', count: 1, type: 'craft'}
 *   "survive and explore"     → null (ongoing, no completion)
 */

// ── Item aliases ──────────────────────────────────────────────────
// Maps goal keywords → actual Minecraft item names in inventory
const ITEM_ALIASES = {
  // Wood
  'wood':        ['oak_log', 'birch_log', 'spruce_log', 'jungle_log', 'acacia_log', 'dark_oak_log'],
  'log':         ['oak_log', 'birch_log', 'spruce_log', 'jungle_log', 'acacia_log', 'dark_oak_log'],
  'logs':        ['oak_log', 'birch_log', 'spruce_log', 'jungle_log', 'acacia_log', 'dark_oak_log'],
  'oak':         ['oak_log', 'oak_planks'],
  'oak_log':     ['oak_log'],
  'oak_logs':    ['oak_log'],
  'birch':       ['birch_log', 'birch_planks'],
  'acacia':      ['acacia_log', 'acacia_planks'],
  'spruce':      ['spruce_log', 'spruce_planks'],
  'plank':       ['oak_planks', 'birch_planks', 'spruce_planks', 'jungle_planks', 'acacia_planks'],
  'planks':      ['oak_planks', 'birch_planks', 'spruce_planks', 'jungle_planks', 'acacia_planks'],
  // Stone
  'stone':       ['cobblestone', 'stone'],
  'cobblestone': ['cobblestone'],
  'cobble':      ['cobblestone'],
  // Ores
  'coal':        ['coal'],
  'iron':        ['raw_iron', 'iron_ingot'],
  'gold':        ['raw_gold', 'gold_ingot'],
  'diamond':     ['diamond'],
  // Food
  'food':        ['bread', 'cooked_beef', 'cooked_porkchop', 'apple', 'carrot', 'potato'],
  'bread':       ['bread'],
  'apple':       ['apple'],
  // Tools
  'wooden_pickaxe': ['wooden_pickaxe'],
  'stone_pickaxe':  ['stone_pickaxe'],
  'iron_pickaxe':   ['iron_pickaxe'],
  'pickaxe':        ['wooden_pickaxe', 'stone_pickaxe', 'iron_pickaxe'],
  'axe':            ['wooden_axe', 'stone_axe', 'iron_axe'],
  'sword':          ['wooden_sword', 'stone_sword', 'iron_sword'],
  'crafting_table': ['crafting_table'],
  'table':          ['crafting_table'],
};

// ── Goal parsing ──────────────────────────────────────────────────

/**
 * Parse a natural-language goal string into a structured goal object.
 * Returns null for open-ended goals with no clear completion condition.
 *
 * @param {string} goalText
 * @returns {{ type: 'collect'|'craft', items: string[], count: number, raw: string } | null}
 */
function parseGoal(goalText) {
  if (!goalText) return null;
  const text = goalText.toLowerCase().trim();

  // ── CRAFT goals ───────────────────────────────────────────────
  // "craft a wooden pickaxe", "craft 2 stone pickaxes", "make a crafting table"
  const craftMatch = text.match(/(?:craft|make|build)\s+(?:a\s+|an\s+)?(?:(\d+)\s+)?([a-z_\s]+?)(?:\s+blocks?)?$/);
  if (craftMatch) {
    const count = parseInt(craftMatch[1] || '1');
    const keyword = craftMatch[2].trim().replace(/\s+/g, '_');
    const items = ITEM_ALIASES[keyword] || ITEM_ALIASES[keyword.replace(/_/g, '')] || [keyword];
    if (items.length > 0) {
      return { type: 'craft', items, count, raw: goalText };
    }
  }

  // ── COLLECT goals ────────────────────────────────────────────
  // "collect 10 wood blocks", "gather 5 oak logs", "mine 3 stone", "get 20 coal"
  const collectMatch = text.match(
    /(?:collect|gather|get|mine|find|obtain|farm)\s+(\d+)\s+([a-z_\s]+?)(?:\s+blocks?|\s+pieces?|\s+units?)?$/
  );
  if (collectMatch) {
    const count = parseInt(collectMatch[1]);
    const keyword = collectMatch[2].trim().replace(/\s+/g, '_');
    // Check direct alias
    const items = ITEM_ALIASES[keyword]
      || ITEM_ALIASES[keyword.replace(/_log$/, '')]
      || ITEM_ALIASES[keyword.replace(/_/g, '')]
      || [keyword.replace(/\s+/g, '_')];

    return { type: 'collect', items, count, raw: goalText };
  }

  // No structured completion — open-ended goal (explore, survive, etc.)
  return null;
}

/**
 * Check how many goal-relevant items the bot currently has.
 *
 * @param {object} inventory - { item_name: count, ... } from state_extractor
 * @param {{ items: string[], count: number }} parsedGoal
 * @returns {{ collected: number, needed: number, done: boolean, progressText: string }}
 */
function checkProgress(inventory, parsedGoal) {
  if (!parsedGoal || !inventory) {
    return { collected: 0, needed: 0, done: false, progressText: '' };
  }

  const collected = parsedGoal.items.reduce((sum, item) => sum + (inventory[item] || 0), 0);
  const needed    = parsedGoal.count;
  const done      = collected >= needed;
  const pct       = Math.min(100, Math.round((collected / needed) * 100));
  const bar       = '█'.repeat(Math.floor(pct / 10)) + '░'.repeat(10 - Math.floor(pct / 10));
  const progressText = `${collected}/${needed} ${parsedGoal.items[0]} [${bar}] ${pct}%`;

  return { collected, needed, done, progressText };
}

/**
 * Build a compact progress string for use in the LLM prompt and logs.
 *
 * @param {object} inventory
 * @param {string} goalText
 * @returns {string} e.g. "Progress: 3/10 oak_log (30%)" or "" if no structured goal
 */
function getProgressText(inventory, goalText) {
  const parsed = parseGoal(goalText);
  if (!parsed) return 'Open-ended goal — no completion condition.';
  const { progressText, done } = checkProgress(inventory, parsed);
  return done ? `GOAL COMPLETE: ${progressText}` : `Progress: ${progressText}`;
}

module.exports = { parseGoal, checkProgress, getProgressText };
