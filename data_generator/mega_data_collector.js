const mineflayer = require('mineflayer');
const { pathfinder } = require('mineflayer-pathfinder');
const fs = require('fs');
const path = require('path');
const { extractState } = require('./extract_state');
const { executeAction } = require('./execute_action');
const { getMegaAction } = require('./mega_agents');

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

// 10 distinct modes
const MODES = [
    "explore", "gather_wood", "gather_stone", "combat", "build", 
    "inventory", "crafting", "eat", "equip", "survive_night"
];
const MAX_STEPS_PER_EPISODE = 100;

const args = process.argv.slice(2);
const port = parseInt(args[0]) || 25565;

const episodesDir = path.join(__dirname, 'episodes');
if (!fs.existsSync(episodesDir)) {
    fs.mkdirSync(episodesDir);
}

function createBotWorker(workerId) {
    let episodeCount = 0;

    function startEpisode() {
        episodeCount++;
        const mode = MODES[Math.floor(Math.random() * MODES.length)];
        const episodeId = Date.now() + "_" + Math.floor(Math.random() * 1000);
        const logFile = path.join(episodesDir, `ep_${episodeId}_${mode}.jsonl`);
        let memory = [];
        let stepCount = 0;
        let isExecuting = false;

        console.log(`\n[Worker_${workerId} | ${mode}] === Starting Episode ${episodeCount}: ${episodeId} ===`);

        const bot = mineflayer.createBot({
            host: 'localhost',
            port: port,
            username: `Worker_${workerId}`,
            version: '1.20.4'
        });

        bot.loadPlugin(pathfinder);

        bot.once('spawn', () => {
            console.log(`[Worker_${workerId} | ${mode}] Spawned. Teleporting to random biome...`);
            // Use 0 0 center to prevent extreme coordinates
            bot.chat(`/spreadplayers 0 0 500 10000 false ${bot.username}`);
            
            setTimeout(() => {
                bot.chat(`/time set day`);
                bot.chat(`/gamemode survival`);
                setTimeout(tickLoop, 3000);
            }, 1000);
        });

        async function tickLoop() {
            if (stepCount >= MAX_STEPS_PER_EPISODE) {
                console.log(`[Worker_${workerId} | ${mode}] Episode complete. Reconnecting...`);
                bot.quit();
                setTimeout(startEpisode, 5000);
                return;
            }

            if (isExecuting) {
                setTimeout(tickLoop, 500);
                return;
            }

            isExecuting = true;
            try {
                const mockGoal = `do_${mode}`; 
                
                const state = extractState(bot, mockGoal, null, memory);
                if (!state) {
                    isExecuting = false;
                    setTimeout(tickLoop, 500);
                    return;
                }

                const action = getMegaAction(state, mode);
                const outcome = await executeAction(bot, action);

                const memStr = `${action.action} ${action.target || action.direction || action.block || action.item || ''} -> ${outcome}`;
                memory.push(memStr);
                if (memory.length > 8) memory.shift();

                const newState = extractState(bot, mockGoal, null, memory);

                const record = {
                    state: state,
                    action: action,
                    next_state: newState
                };
                fs.appendFileSync(logFile, canonical(record) + '\n');
                
                stepCount++;
            } catch (e) {
                console.error(`[Worker_${workerId} | ${mode}] Tick error:`, e.message);
            }

            isExecuting = false;
            setTimeout(tickLoop, 500);
        }

        bot.on('error', (err) => {
            console.log(`[Worker_${workerId}] Error:`, err.message);
            bot.quit();
            setTimeout(startEpisode, 10000);
        });
        
        bot.on('kicked', (reason) => {
            console.log(`[Worker_${workerId}] Kicked:`, reason);
            setTimeout(startEpisode, 10000);
        });
    }

    startEpisode();
}

// Start only 2 concurrent workers to prevent server crash
const NUM_CONCURRENT_BOTS = 2;

for (let i = 0; i < NUM_CONCURRENT_BOTS; i++) {
    setTimeout(() => {
        createBotWorker(i);
    }, i * 5000); // Stagger joins by 5 seconds
}
