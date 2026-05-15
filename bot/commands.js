/**
 * commands.js
 * Registers in-game chat commands that players can type.
 * Day 1: rule-based. Phase 2: these will delegate to the LLM planner.
 */

const { extractState } = require('./state_extractor');

const COMMANDS = {
  '!help':      'Show this list',
  '!status':    'Health, food, position',
  '!state':     'Print full Observation JSON',
  '!inventory': 'Show inventory',
  '!follow':    'Bot follows you',
  '!stop':      'Stop all movement',
  '!goto x y z': 'Walk to coordinates',
  '!mine <block>': 'Mine nearest named block',
  '!say <msg>': 'Bot says message',
};

function setupCommands(bot, actions) {
  bot.on('chat', (username, message) => {
    // Ignore own messages
    if (username === bot.username) return;

    const raw  = message.trim();
    const cmd  = raw.toLowerCase();
    const args = raw.split(' ').slice(1);  // parts after the command word

    if (cmd === '!help')        return handleHelp(bot);
    if (cmd === '!status')      return handleStatus(bot);
    if (cmd === '!state')       return handleState(bot);
    if (cmd === '!inventory')   return handleInventory(bot);
    if (cmd === '!follow')      return handleFollow(bot, actions, username);
    if (cmd === '!stop')        return handleStop(bot, actions);
    if (cmd.startsWith('!goto '))  return handleGoto(bot, actions, args);
    if (cmd.startsWith('!mine '))  return handleMine(bot, actions, args);
    if (cmd.startsWith('!say '))   return bot.chat(raw.slice(5));
  });
}

// ─── Handlers ────────────────────────────────────────────────────

function handleHelp(bot) {
  bot.chat('=== MineAgent v0.1 ===');
  for (const [cmd, desc] of Object.entries(COMMANDS)) {
    bot.chat(`${cmd} — ${desc}`);
  }
}

function handleStatus(bot) {
  const pos = bot.entity?.position;
  const px  = pos ? Math.floor(pos.x) : '?';
  const py  = pos ? Math.floor(pos.y) : '?';
  const pz  = pos ? Math.floor(pos.z) : '?';
  bot.chat(`❤ ${bot.health}/20  🍗 ${bot.food}/20  📍 (${px}, ${py}, ${pz})`);
  bot.chat(`🕐 ${bot.time?.timeOfDay ?? '?'}  🌦 ${bot.isRaining ? 'Rain' : 'Clear'}`);
}

function handleState(bot) {
  const state = extractState(bot);
  const json  = JSON.stringify(state, null, 2);
  // Minecraft chat lines max ~256 chars; send first 4 lines as a teaser
  const lines = json.split('\n').slice(0, 6);
  lines.forEach(l => bot.chat(l));
  bot.chat(`... (full state logged to console)`);
  console.log('[STATE]', json);
}

function handleInventory(bot) {
  const items = bot.inventory.items();
  if (items.length === 0) { bot.chat('Inventory is empty.'); return; }
  const summary = items.map(i => `${i.name}×${i.count}`).join(', ');
  bot.chat(`🎒 ${summary}`);
}

function handleFollow(bot, actions, username) {
  const target = bot.players[username]?.entity;
  if (!target) { bot.chat(`Cannot find player: ${username}`); return; }
  actions.follow(target);
  bot.chat(`Following ${username}!`);
}

function handleStop(bot, actions) {
  actions.stopMoving();
  bot.chat('Stopped. ✋');
}

function handleGoto(bot, actions, args) {
  if (args.length < 3) { bot.chat('Usage: !goto x y z'); return; }
  const [x, y, z] = args.map(Number);
  if ([x, y, z].some(isNaN)) { bot.chat('Invalid coordinates.'); return; }
  actions.goTo(x, y, z);
  bot.chat(`🚶 Walking to (${x}, ${y}, ${z})…`);
}

function handleMine(bot, actions, args) {
  if (args.length < 1) { bot.chat('Usage: !mine <block_name>'); return; }
  const blockName = args[0].toLowerCase();
  bot.chat(`⛏ Mining nearest ${blockName}…`);
  actions.mineNearest(blockName)
    .then(ok => bot.chat(ok ? `Mined ${blockName}!` : `Could not mine ${blockName}.`))
    .catch(err => bot.chat(`Error: ${err.message}`));
}

module.exports = { setupCommands };
