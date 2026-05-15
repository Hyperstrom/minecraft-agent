require('dotenv').config({ path: '../.env' });

module.exports = {
  // Minecraft Server
  host:     process.env.MC_HOST     || 'localhost',
  port:     parseInt(process.env.MC_PORT) || 25565,
  username: process.env.MC_USERNAME || 'MineAgent',
  version:  process.env.MC_VERSION  || '1.20.4',
  auth:     'offline',

  // Backend API
  backendUrl: process.env.BACKEND_URL || 'http://localhost:8000',

  // Agent behaviour
  stateLogInterval: 30000,   // ms between state logs
  reconnectDelay:    5000,   // ms before reconnecting after disconnect
};
