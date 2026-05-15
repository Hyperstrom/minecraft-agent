/**
 * logger.js
 * Simple timestamped console logger with coloured level tags.
 */

const LEVELS = {
  info:  '\x1b[36m[INFO] \x1b[0m',   // cyan
  warn:  '\x1b[33m[WARN] \x1b[0m',   // yellow
  error: '\x1b[31m[ERROR]\x1b[0m',   // red
  state: '\x1b[35m[STATE]\x1b[0m',   // magenta
  chat:  '\x1b[32m[CHAT] \x1b[0m',   // green
};

function ts() {
  return new Date().toISOString().replace('T', ' ').split('.')[0];
}

const log = {
  info:  (msg) => console.log(`${ts()} ${LEVELS.info}  ${msg}`),
  warn:  (msg) => console.warn(`${ts()} ${LEVELS.warn}  ${msg}`),
  error: (msg) => console.error(`${ts()} ${LEVELS.error} ${msg}`),
  state: (msg) => console.log(`${ts()} ${LEVELS.state} ${msg}`),
  chat:  (msg) => console.log(`${ts()} ${LEVELS.chat}  ${msg}`),
};

module.exports = log;
