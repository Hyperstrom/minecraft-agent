function getGreedyAction(state) {
    if (state.nearby && state.nearby.length > 0) {
        // Try to mine the closest block that isn't dirt/grass if possible, or just the closest
        let target = state.nearby[0];
        for (let b of state.nearby) {
            if (b.name.includes('log') || b.name.includes('ore') || b.name.includes('stone')) {
                target = b;
                break;
            }
        }
        return { action: "MINE", target: target.name };
    }
    return { action: "MOVE", direction: "north", distance: 5 };
}

function getCautiousAction(state) {
    if (state.hp < 10 || (state.mobs && state.mobs.length > 0 && state.mobs[0].dist < 5)) {
        // Flee
        return { action: "MOVE", direction: "south", distance: 10 };
    }
    return getGreedyAction(state); // fallback
}

function getExplorerAction(state) {
    const dirs = ["north", "south", "east", "west"];
    const randomDir = dirs[Math.floor(Math.random() * dirs.length)];
    return { action: "MOVE", direction: randomDir, distance: 8 };
}

function getDumbAction(state) {
    // Always mine closest log, otherwise wait
    if (state.nearby) {
        const log = state.nearby.find(b => b.name.includes('log'));
        if (log) return { action: "MINE", target: log.name };
    }
    return { action: "WAIT", ticks: 20 };
}

module.exports = {
    getGreedyAction,
    getCautiousAction,
    getExplorerAction,
    getDumbAction
};
