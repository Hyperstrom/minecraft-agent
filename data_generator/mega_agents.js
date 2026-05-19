function getExploreAction(state) {
    const dirs = ["north", "south", "east", "west"];
    const randomDir = dirs[Math.floor(Math.random() * dirs.length)];
    return { action: "MOVE", direction: randomDir, distance: Math.floor(Math.random() * 5) + 3 };
}

function isStuck(state) {
    if (!state.memory || state.memory.length < 2) return false;
    const last = state.memory[state.memory.length - 1];
    
    // If last action failed, we are stuck
    if (last.includes("failed")) return true;
    
    // If we repeated the exact same action 4 times recently (e.g. mining same block but not picking it up)
    const actionPattern = last.split(" -> ")[0];
    let count = 0;
    for (let m of state.memory) {
        if (m.startsWith(actionPattern)) count++;
    }
    if (count >= 4) return true;
    
    return false;
}

function getGatherWoodAction(state) {
    if (state.nearby && state.nearby.length > 0) {
        let target = state.nearby.find(b => b.name.includes('log'));
        if (target) return { action: "MINE", target: target.name };
    }
    return getExploreAction(state);
}

function getGatherStoneAction(state) {
    if (state.nearby && state.nearby.length > 0) {
        let target = state.nearby.find(b => b.name.includes('stone') || b.name.includes('coal') || b.name.includes('ore'));
        if (target) return { action: "MINE", target: target.name };
    }
    return getExploreAction(state);
}

function getCombatAction(state) {
    if (state.hp < 8) {
        return { action: "MOVE", direction: "south", distance: 10 };
    }
    if (state.inv && (state.inv['iron_sword'] || state.inv['wooden_sword']) && !state.equipped?.includes('sword')) {
        const sword = Object.keys(state.inv).find(i => i.includes('sword'));
        if (sword) return { action: "EQUIP", item: sword, slot: "hand" };
    }
    if (state.mobs && state.mobs.length > 0) {
        return { action: "ATTACK", target: state.mobs[0].name };
    }
    return getExploreAction(state);
}

function getBuildAction(state) {
    if (state.inv) {
        const placeable = Object.keys(state.inv).find(item => item.includes('cobblestone') || item.includes('planks') || item.includes('dirt') || item.includes('log'));
        if (placeable) {
            const pos = { x: state.pos.x + 1, y: state.pos.y, z: state.pos.z };
            return { action: "PLACE", block: placeable, pos: pos };
        }
    }
    // If no blocks to build, gather wood to build with
    return getGatherWoodAction(state);
}

function getInventoryAction(state) {
    if (state.inv) {
        const items = Object.keys(state.inv);
        if (items.length > 3) {
            return { action: "DROP", item: items[0], quantity: 1 };
        }
    }
    return getExploreAction(state);
}

function getCraftingAction(state) {
    const inv = state.inv || {};
    const logs = Object.keys(inv).find(i => i.includes('log'));
    const planks = Object.keys(inv).find(i => i.includes('planks'));
    const sticks = inv['stick'];
    const table = inv['crafting_table'];
    
    // Craft planks
    if (logs && (!planks || inv[planks] < 4)) {
        return { action: "CRAFT", target: "oak_planks", quantity: 1 };
    }
    // Craft stick
    if (planks && !sticks) {
        return { action: "CRAFT", target: "stick", quantity: 1 };
    }
    // Craft crafting table
    if (planks && inv[planks] >= 4 && !table) {
        return { action: "CRAFT", target: "crafting_table", quantity: 1 };
    }
    // Craft wooden pickaxe (needs sticks and planks, doesn't actually need table in inventory if it's placed, but this is simple logic)
    if (planks && inv[planks] >= 3 && sticks && inv['stick'] >= 2 && !inv['wooden_pickaxe']) {
        return { action: "CRAFT", target: "wooden_pickaxe", quantity: 1 };
    }
    
    // If we have nothing to craft, go get wood
    return getGatherWoodAction(state);
}

function getEatAction(state) {
    if (state.food < 15 || state.hp < 15) {
        if (state.inv) {
            const food = Object.keys(state.inv).find(i => i.includes('apple') || i.includes('bread') || i.includes('beef') || i.includes('porkchop') || i.includes('mutton') || i.includes('chicken'));
            if (food) {
                return { action: "EAT", item: food };
            }
        }
    }
    if (state.mobs && state.mobs.length > 0) {
        const animal = state.mobs.find(m => !m.hostile);
        if (animal) return { action: "ATTACK", target: animal.name };
    }
    return getExploreAction(state);
}

function getEquipAction(state) {
    if (state.inv) {
        const armor = Object.keys(state.inv).find(i => i.includes('helmet') || i.includes('chestplate') || i.includes('leggings') || i.includes('boots'));
        if (armor) {
            let slot = 'head';
            if (armor.includes('chestplate')) slot = 'chest';
            if (armor.includes('leggings')) slot = 'legs';
            if (armor.includes('boots')) slot = 'feet';
            return { action: "EQUIP", item: armor, slot: slot };
        }
        const tool = Object.keys(state.inv).find(i => i.includes('pickaxe') || i.includes('axe') || i.includes('sword'));
        if (tool && state.equipped !== tool) {
            return { action: "EQUIP", item: tool, slot: "hand" };
        }
    }
    return getGatherWoodAction(state);
}

function getSurviveNightAction(state) {
    if (state.time > 12000 && state.time < 23000) {
        if (state.mobs && state.mobs.length > 0) {
            const threat = state.mobs.find(m => m.hostile && m.dist < 8);
            if (threat) return { action: "MOVE", direction: "south", distance: 15 };
        }
        return { action: "WAIT", ticks: 100 };
    }
    return getExploreAction(state);
}

function getMegaAction(state, mode) {
    // ANTI-LOOP CHECK: If stuck, break the loop by moving randomly
    if (isStuck(state)) {
        return getExploreAction(state);
    }

    switch (mode) {
        case "explore": return getExploreAction(state);
        case "gather_wood": return getGatherWoodAction(state);
        case "gather_stone": return getGatherStoneAction(state);
        case "combat": return getCombatAction(state);
        case "build": return getBuildAction(state);
        case "inventory": return getInventoryAction(state);
        case "crafting": return getCraftingAction(state);
        case "eat": return getEatAction(state);
        case "equip": return getEquipAction(state);
        case "survive_night": return getSurviveNightAction(state);
        default: return getExploreAction(state);
    }
}

module.exports = { getMegaAction };
