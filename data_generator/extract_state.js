const mineflayer = require('mineflayer');

function extractState(bot, currentGoal = null, currentSubgoal = null, memory = []) {
    if (!bot || !bot.entity) return null;

    const pos = bot.entity.position;
    
    // Calculate facing direction
    let facing = 'north';
    const yaw = bot.entity.yaw;
    const PI = Math.PI;
    if (yaw >= -PI/4 && yaw < PI/4) facing = 'south';
    else if (yaw >= PI/4 && yaw < 3*PI/4) facing = 'west';
    else if (yaw >= -3*PI/4 && yaw < -PI/4) facing = 'east';

    // Inventory
    const inv = {};
    for (const item of bot.inventory.items()) {
        inv[item.name] = (inv[item.name] || 0) + item.count;
    }

    const equipped = bot.heldItem ? bot.heldItem.name : null;

    // Nearby blocks (find visible blocks)
    const nearby = [];
    const blockNamesSet = new Set();
    const radius = 8;
    for (let x = -radius; x <= radius; x++) {
        for (let y = -radius; y <= radius; y++) {
            for (let z = -radius; z <= radius; z++) {
                const b = bot.blockAt(pos.offset(x, y, z));
                if (b && b.name !== 'air' && b.name !== 'cave_air' && b.name !== 'water' && b.name !== 'bedrock') {
                    const dist = pos.distanceTo(b.position);
                    if (dist <= radius) {
                        nearby.push({
                            name: b.name,
                            dist: parseFloat(dist.toFixed(1))
                        });
                    }
                }
            }
        }
    }
    
    // Sort nearby by distance, filter out duplicates roughly, cap at 8
    nearby.sort((a, b) => a.dist - b.dist);
    const uniqueNearby = [];
    for (const b of nearby) {
        if (!blockNamesSet.has(b.name)) {
            blockNamesSet.add(b.name);
            uniqueNearby.push(b);
            if (uniqueNearby.length >= 8) break;
        }
    }

    // Mobs
    const mobs = [];
    for (const id in bot.entities) {
        const entity = bot.entities[id];
        if (entity === bot.entity) continue;
        if (entity.type === 'mob' || entity.type === 'animal') {
            const dist = pos.distanceTo(entity.position);
            if (dist <= 16) {
                mobs.push({
                    name: entity.name,
                    dist: parseFloat(dist.toFixed(1)),
                    hp: entity.health || 20,
                    hostile: entity.kind === 'Hostile mobs'
                });
            }
        }
    }
    mobs.sort((a, b) => a.dist - b.dist);
    mobs.splice(4); // Cap at 4

    // Biome
    let biome = 'plains'; // fallback
    try {
        const block = bot.blockAt(pos);
        if (block && block.biome && block.biome.name) {
            biome = block.biome.name;
        } else {
            // Mineflayer 1.20 workaround if biome.name is empty
            const bId = bot.world.getBiome(pos);
            const mcData = require('minecraft-data')(bot.version);
            if (bId !== undefined && mcData.biomes[bId]) {
                biome = mcData.biomes[bId].name;
            }
        }
    } catch(e) {}
    
    if (!biome || biome === "") biome = "plains";

    return {
        v: 1,
        hp: bot.health || 20,
        food: bot.food || 20,
        pos: { x: Math.floor(pos.x), y: Math.floor(pos.y), z: Math.floor(pos.z) },
        facing: facing,
        inv: inv,
        equipped: equipped,
        nearby: uniqueNearby,
        mobs: mobs,
        time: bot.time.timeOfDay,
        raining: bot.isRaining,
        biome: biome,
        goal: currentGoal,
        subgoal: currentSubgoal,
        memory: memory
    };
}

module.exports = { extractState };
