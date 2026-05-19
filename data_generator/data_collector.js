const mineflayer = require('mineflayer');
const { pathfinder } = require('mineflayer-pathfinder');
const fs = require('fs');
const path = require('path');
const { extractState } = require('./extract_state');
const { executeAction } = require('./execute_action');
const agents = require('./dumb_agents');

// Ensure canonical JSON format
function canonical(obj) {
    if (Array.isArray(obj)) {
        return `[${obj.map(canonical).join(',')}]`;
    } else if (obj !== null && typeof obj === 'object') {
        const keys = Object.keys(obj).sort();
        const props = keys.map(k => `"${k}":${canonical(obj[k])}`);
        return `{${props.join(',')}}`;
    } else {
        return JSON.stringify(obj);
    }
}

const args = process.argv.slice(2);
const agentType = args[0] || 'greedy'; // greedy, cautious, explorer, dumb
const getAction = agents[`get${agentType.charAt(0).toUpperCase() + agentType.slice(1)}Action`] || agents.getGreedyAction;

const bot = mineflayer.createBot({
    host: 'localhost',
    port: 25565,
    username: `DataBot_${agentType}`,
    version: '1.20.4'
});

bot.loadPlugin(pathfinder);

let memory = [];
let episodeId = Date.now();
const logFile = path.join(__dirname, 'episodes', `ep_${episodeId}_${agentType}.jsonl`);
let stepCount = 0;
const MAX_STEPS = 50;
let isExecuting = false;

bot.once('spawn', () => {
    console.log(`Bot ${agentType} spawned. Starting episode ${episodeId}`);
    
    // Give bot time to load chunks
    setTimeout(tickLoop, 2000);
});

async function tickLoop() {
    if (stepCount >= MAX_STEPS) {
        console.log("Episode complete. Exiting.");
        bot.quit();
        process.exit(0);
        return;
    }

    if (isExecuting) {
        setTimeout(tickLoop, 500);
        return;
    }

    isExecuting = true;
    try {
        const state = extractState(bot, "chop_wood", null, memory);
        if (!state) {
            isExecuting = false;
            setTimeout(tickLoop, 500);
            return;
        }

        const action = getAction(state);
        
        // Execute Action
        console.log(`[Step ${stepCount}] Action: ${action.action} ${action.target || action.direction || ''}`);
        const outcome = await executeAction(bot, action);
        console.log(`   -> Outcome: ${outcome}`);

        // Update memory
        const memStr = `${action.action} ${action.target || action.direction || ''} -> ${outcome}`;
        memory.push(memStr);
        if (memory.length > 8) memory.shift();

        // Get new state
        const newState = extractState(bot, "chop_wood", null, memory);

        // Log to JSONL
        const record = {
            state: state,
            action: action,
            next_state: newState
        };
        fs.appendFileSync(logFile, canonical(record) + '\n');
        
        stepCount++;
    } catch (e) {
        console.error("Tick error:", e);
    }

    isExecuting = false;
    setTimeout(tickLoop, 500);
}

bot.on('error', console.log);
bot.on('kicked', console.log);
