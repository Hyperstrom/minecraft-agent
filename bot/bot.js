/**
 * bot.js  —  MineAgent Phase 2 entry point
 *
 * New in Phase 2:
 *  - Auto-planning loop (planner_client) starts on spawn
 *  - !planner on/off — toggle autonomous planning
 *  - !goal <text>    — set the agent's current goal
 *  - !status shows planner state + current goal
 */

require('dotenv').config({ path: '../.env' });

const mineflayer   = require('mineflayer');
const { pathfinder } = require('mineflayer-pathfinder');

const config          = require('./config');
const log             = require('./logger');
const { extractState }  = require('./state_extractor');
const { setupActions }  = require('./actions');
const { setupCommands } = require('./commands');
const planner           = require('./planner_client');

// ── Bot factory ───────────────────────────────────────────────────
function createBot() {
  log.info(`Connecting to ${config.host}:${config.port} as "${config.username}"`);

  const bot = mineflayer.createBot({
    host:     config.host,
    port:     config.port,
    username: config.username,
    version:  config.version,
    auth:     config.auth,
  });

  // Raise limit to avoid "MaxListenersExceededWarning" from repeated goNear calls
  bot.setMaxListeners(50);

  bot.loadPlugin(pathfinder);


  // ── Spawn ──────────────────────────────────────────────────────
  bot.once('spawn', () => {
    log.info('Bot spawned!');

    const actions = setupActions(bot);
    setupCommands(bot, actions, planner);  // planner passed — handles all commands

    setTimeout(() => {
      const s = extractState(bot);
      log.state(JSON.stringify(s, null, 2));
      bot.chat('MineAgent online! Use !goal <text> to set a goal, then !planner on to start.');

      // Spawn near the first real player found
      const players = Object.values(bot.players).filter(p => p.username !== bot.username && p.entity);
      if (players.length > 0) {
        const pos = players[0].entity.position;
        const actions = setupActions(bot);
        actions.goNear(pos.x, pos.y, pos.z, 4);
        log.info(`[Bot] Moving near player: ${players[0].username}`);
      }

      // DO NOT set a goal or start planner automatically
      // User must type: !goal <text>  then  !planner on
    }, 1500);

    const hb = setInterval(() => {
      const s = extractState(bot);
      log.state(
        `HP:${s.player.health} Food:${s.player.food} ` +
        `Pos:(${s.player.position?.x},${s.player.position?.y},${s.player.position?.z}) ` +
        `Goal:"${planner.getGoal()}"`
      );
    }, config.stateLogInterval);

    bot.once('end', () => { clearInterval(hb); planner.stop(); });
  });

  // ── Health warning ─────────────────────────────────────────────
  bot.on('health', () => {
    if (bot.health !== undefined && bot.health <= 6) log.warn(`Low HP: ${bot.health}/20`);
  });

  bot.on('death',  () => { log.warn('Bot died — respawning'); bot.respawn(); });
  bot.on('chat',   (u, m) => { if (u !== bot.username) log.chat(`<${u}> ${m}`); });
  bot.on('playerJoined', p => log.info(`+ ${p.username} joined`));
  bot.on('playerLeft',   p => log.info(`- ${p.username} left`));
  bot.on('kicked', reason => { log.error(`Kicked: ${reason}`); setTimeout(createBot, config.reconnectDelay); });
  bot.on('error',  err    => log.error(`Bot error: ${err.message}`));
  bot.on('end',    reason => {
    log.warn(`Disconnected (${reason}). Reconnecting in ${config.reconnectDelay / 1000}s…`);
    setTimeout(createBot, config.reconnectDelay);
  });

  return bot;
}

// ── Entry point ───────────────────────────────────────────────────
createBot();
