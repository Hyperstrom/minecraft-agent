/**
 * test_state.js  —  Day 1 Unit Tests
 * Run with:  node tests/test_state.js
 *
 * Tests all pure functions in state_extractor.js using a mock bot.
 * No live Minecraft server required.
 */

const assert = require('assert');
const {
  extractState,
  getInventory,
  getNearbyEntities,
  getNearbyBlocks,
  getTimeOfDay,
  roundPosition,
} = require('../state_extractor');

// ─── Mock Bot ────────────────────────────────────────────────────

function makeVec3(x, y, z) {
  return {
    x, y, z,
    floored: () => makeVec3(Math.floor(x), Math.floor(y), Math.floor(z)),
    offset:  (dx, dy, dz) => makeVec3(x + dx, y + dy, z + dz),
    distanceTo: (other) => Math.sqrt(
      (x - other.x) ** 2 + (y - other.y) ** 2 + (z - other.z) ** 2
    ),
  };
}

const MOCK_BOT = {
  username: 'MineAgent',
  health:   18,
  food:     16,
  foodSaturation: 4,
  isRaining: false,
  time: { timeOfDay: 8000 },  // noon

  entity: {
    position: makeVec3(100.5, 64.0, -22.3),
  },

  entities: {
    'e1': {
      type: 'player',
      name: null,
      username: 'TestPlayer',
      position: makeVec3(103.0, 64.0, -22.3),
    },
    'e2': {
      type: 'mob',
      name: 'zombie',
      position: makeVec3(110.0, 64.0, -22.3),
    },
    'self': null,  // skipped (no position)
  },

  inventory: {
    items: () => [
      { name: 'oak_log',  count: 5 },
      { name: 'iron_ore', count: 3 },
      { name: 'oak_log',  count: 2 },  // duplicate — should be merged
    ],
  },

  // No interesting blocks nearby → returns []
  blockAt: () => null,
};

// Self-reference for entity filtering
MOCK_BOT.entities['self'] = MOCK_BOT.entity;

// ─── Test runner ─────────────────────────────────────────────────

let passed = 0;
let failed = 0;

function test(name, fn) {
  try {
    fn();
    console.log(`  ✅  ${name}`);
    passed++;
  } catch (err) {
    console.error(`  ❌  ${name}\n      → ${err.message}`);
    failed++;
  }
}

// ─── Tests ───────────────────────────────────────────────────────

console.log('\n🧪  MineAgent Day-1 Unit Tests\n');

// 1. roundPosition
test('roundPosition — null input returns null', () => {
  assert.strictEqual(roundPosition(null), null);
});

test('roundPosition — rounds to 1 decimal', () => {
  const r = roundPosition({ x: 1.123, y: 64.999, z: -22.351 });
  assert.strictEqual(r.x, 1.1);
  assert.strictEqual(r.y, 65.0);
  assert.strictEqual(r.z, -22.4);
});

// 2. getInventory
test('getInventory — aggregates duplicate items', () => {
  const inv = getInventory(MOCK_BOT);
  assert.strictEqual(inv['oak_log'],  7, 'oak_log: 5+2 = 7');
  assert.strictEqual(inv['iron_ore'], 3);
});

test('getInventory — returns plain object', () => {
  const inv = getInventory(MOCK_BOT);
  assert.strictEqual(typeof inv, 'object');
});

// 3. getNearbyEntities
test('getNearbyEntities — skips self entity', () => {
  const entities = getNearbyEntities(MOCK_BOT, 64);
  const self = entities.find(e => e.name === 'MineAgent');
  assert.strictEqual(self, undefined);
});

test('getNearbyEntities — finds TestPlayer', () => {
  const entities = getNearbyEntities(MOCK_BOT, 64);
  const player = entities.find(e => e.name === 'TestPlayer');
  assert.ok(player, 'TestPlayer should be found');
  assert.ok(player.distance > 0, 'distance should be > 0');
});

test('getNearbyEntities — sorted by distance ascending', () => {
  const entities = getNearbyEntities(MOCK_BOT, 64);
  for (let i = 1; i < entities.length; i++) {
    assert.ok(entities[i].distance >= entities[i - 1].distance,
      'entities should be sorted by distance');
  }
});

test('getNearbyEntities — max 10 results', () => {
  const entities = getNearbyEntities(MOCK_BOT, 64);
  assert.ok(entities.length <= 10);
});

// 4. getTimeOfDay
test('getTimeOfDay — morning (tick 1000)', () => {
  assert.strictEqual(getTimeOfDay({ time: { timeOfDay: 1000 } }), 'morning');
});
test('getTimeOfDay — noon (tick 8000)', () => {
  assert.strictEqual(getTimeOfDay({ time: { timeOfDay: 8000 } }), 'noon');
});
test('getTimeOfDay — sunset (tick 12500)', () => {
  assert.strictEqual(getTimeOfDay({ time: { timeOfDay: 12500 } }), 'sunset');
});
test('getTimeOfDay — night (tick 14000)', () => {
  assert.strictEqual(getTimeOfDay({ time: { timeOfDay: 14000 } }), 'night');
});

// 5. extractState — schema validation
test('extractState — returns all top-level keys', () => {
  const state = extractState(MOCK_BOT, 'gather wood');
  const required = ['player', 'inventory', 'nearby_entities', 'nearby_blocks', 'environment', 'goal', 'timestamp'];
  for (const key of required) {
    assert.ok(key in state, `Missing key: ${key}`);
  }
});

test('extractState — player sub-keys present', () => {
  const { player } = extractState(MOCK_BOT);
  for (const key of ['username', 'position', 'health', 'food']) {
    assert.ok(key in player, `Missing player.${key}`);
  }
});

test('extractState — player values correct', () => {
  const { player } = extractState(MOCK_BOT);
  assert.strictEqual(player.username, 'MineAgent');
  assert.strictEqual(player.health, 18);
  assert.strictEqual(player.food, 16);
});

test('extractState — goal is passed through', () => {
  const state = extractState(MOCK_BOT, 'gather wood');
  assert.strictEqual(state.goal, 'gather wood');
});

test('extractState — goal null when omitted', () => {
  const state = extractState(MOCK_BOT);
  assert.strictEqual(state.goal, null);
});

test('extractState — timestamp is valid ISO string', () => {
  const { timestamp } = extractState(MOCK_BOT);
  assert.ok(!isNaN(Date.parse(timestamp)), 'timestamp must be ISO date');
});

test('extractState — weather is clear', () => {
  assert.strictEqual(extractState(MOCK_BOT).environment.weather, 'clear');
});

test('extractState — weather is rain when isRaining=true', () => {
  const rainyBot = { ...MOCK_BOT, isRaining: true };
  assert.strictEqual(extractState(rainyBot).environment.weather, 'rain');
});

// 6. JSON serialisation
test('extractState — output is fully JSON-serialisable', () => {
  const state = extractState(MOCK_BOT);
  let json;
  try {
    json = JSON.stringify(state);
    JSON.parse(json);
  } catch (e) {
    assert.fail(`State is not JSON-safe: ${e.message}`);
  }
  assert.ok(json.length > 50, 'JSON output should be non-trivial');
});

// ─── Summary ─────────────────────────────────────────────────────

console.log(`\n📊  Results: ${passed} passed, ${failed} failed\n`);
if (failed > 0) process.exit(1);
