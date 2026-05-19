const { pathfinder, Movements, goals: { GoalNear, GoalBlock, GoalXZ } } = require('mineflayer-pathfinder');
const Vec3 = require('vec3');

async function executeAction(bot, actionObj) {
    if (!actionObj || !actionObj.action) {
        return "failed: invalid action";
    }

    const mcData = require('minecraft-data')(bot.version);
    const movements = new Movements(bot, mcData);
    movements.canDig = true;
    movements.allowFreeClear = true;
    bot.pathfinder.setMovements(movements);

    try {
        switch (actionObj.action) {
            case "MINE": {
                if (!actionObj.target) return "failed: missing target";
                const blockName = actionObj.target;
                const block = bot.findBlock({ matching: mcData.blocksByName[blockName]?.id, maxDistance: 32 });
                if (!block) return `failed: ${blockName} not found`;
                
                await bot.pathfinder.goto(new GoalNear(block.position.x, block.position.y, block.position.z, 1));
                if (bot.canDigBlock(block)) {
                    await bot.dig(block);
                    await new Promise(r => setTimeout(r, 500));
                    const drop = Object.values(bot.entities).find(e => 
                        e.type === 'object' && 
                        e.position.distanceTo(block.position) < 4
                    );
                    if (drop) {
                        try { await bot.pathfinder.goto(new GoalNear(drop.position.x, drop.position.y, drop.position.z, 0.5)); } catch(e) {}
                    } else {
                        try { await bot.pathfinder.goto(new GoalNear(block.position.x, block.position.y, block.position.z, 0)); } catch(e) {}
                    }
                    return `success: mined ${blockName}`;
                }
                return `failed: cannot dig ${blockName}`;
            }
            case "MOVE": {
                if (!actionObj.direction || !actionObj.distance) return "failed: missing args";
                let dx=0, dz=0;
                if (actionObj.direction === 'north') dz = -actionObj.distance;
                if (actionObj.direction === 'south') dz = actionObj.distance;
                if (actionObj.direction === 'east') dx = actionObj.distance;
                if (actionObj.direction === 'west') dx = -actionObj.distance;
                
                const targetPos = bot.entity.position.offset(dx, 0, dz);
                await bot.pathfinder.goto(new GoalXZ(targetPos.x, targetPos.z));
                return `success: moved ${actionObj.direction} ${actionObj.distance}`;
            }
            case "MOVE_TO": {
                if (!actionObj.pos) return "failed: missing pos";
                await bot.pathfinder.goto(new GoalNear(actionObj.pos.x, actionObj.pos.y, actionObj.pos.z, 1));
                return `success: moved to ${actionObj.pos.x},${actionObj.pos.y},${actionObj.pos.z}`;
            }
            case "EAT": {
                if (!actionObj.item) return "failed: missing food item";
                const item = bot.inventory.items().find(i => i.name === actionObj.item);
                if (!item) return `failed: no ${actionObj.item} in inventory`;
                await bot.equip(item, 'hand');
                await bot.consume();
                return `success: ate ${actionObj.item}`;
            }
            case "CRAFT": {
                if (!actionObj.target) return "failed: missing target to craft";
                const itemName = actionObj.target;
                const qty = actionObj.quantity || 1;
                const itemType = mcData.itemsByName[itemName];
                if (!itemType) return `failed: unknown item ${itemName}`;
                
                const recipe = bot.recipesFor(itemType.id, null, 1, null)[0];
                if (!recipe) return `failed: no recipe or missing ingredients for ${itemName}`;
                
                let craftingTable = null;
                if (recipe.requiresTable) {
                    craftingTable = bot.findBlock({ matching: mcData.blocksByName['crafting_table'].id, maxDistance: 32 });
                    if (!craftingTable) return `failed: requires crafting_table nearby`;
                    await bot.pathfinder.goto(new GoalNear(craftingTable.position.x, craftingTable.position.y, craftingTable.position.z, 2));
                }
                
                await bot.craft(recipe, qty, craftingTable);
                return `success: crafted ${qty} ${itemName}`;
            }
            case "EQUIP": {
                if (!actionObj.item || !actionObj.slot) return "failed: missing item or slot";
                const item = bot.inventory.items().find(i => i.name === actionObj.item);
                if (!item) return `failed: no ${actionObj.item} in inventory`;
                
                let dest = actionObj.slot; // hand, head, chest, legs, feet
                if (dest === 'hand') dest = 'hand';
                else if (dest === 'head') dest = 'head';
                else if (dest === 'chest') dest = 'torso';
                else if (dest === 'legs') dest = 'legs';
                else if (dest === 'feet') dest = 'feet';
                
                await bot.equip(item, dest);
                return `success: equipped ${actionObj.item} to ${actionObj.slot}`;
            }
            case "PLACE": {
                if (!actionObj.block || !actionObj.pos) return "failed: missing block or pos";
                const item = bot.inventory.items().find(i => i.name === actionObj.block);
                if (!item) return `failed: no ${actionObj.block} in inventory`;
                
                await bot.equip(item, 'hand');
                
                let referenceBlock = null;
                const radius = 2;
                for (let x = -radius; x <= radius; x++) {
                    for (let z = -radius; z <= radius; z++) {
                        const b = bot.blockAt(bot.entity.position.offset(x, -1, z));
                        const above = bot.blockAt(bot.entity.position.offset(x, 0, z));
                        if (b && b.name !== 'air' && above && above.name === 'air') {
                            referenceBlock = b;
                            break;
                        }
                    }
                    if (referenceBlock) break;
                }
                
                if (referenceBlock) {
                    try {
                        await bot.placeBlock(referenceBlock, new Vec3(0, 1, 0));
                        return `success: placed ${actionObj.block}`;
                    } catch(e) {
                        return `failed: ${e.message}`;
                    }
                }
                return "failed: no reference block to place on";
            }
            case "ATTACK": {
                if (!actionObj.target) return "failed: missing target entity";
                const entity = Object.values(bot.entities).find(e => e.name === actionObj.target && e !== bot.entity);
                if (!entity) return `failed: entity ${actionObj.target} not found`;
                
                await bot.pathfinder.goto(new GoalNear(entity.position.x, entity.position.y, entity.position.z, 2));
                bot.attack(entity);
                return `success: attacked ${actionObj.target}`;
            }
            case "DROP": {
                if (!actionObj.item) return "failed: missing item to drop";
                const item = bot.inventory.items().find(i => i.name === actionObj.item);
                if (!item) return `failed: no ${actionObj.item} in inventory`;
                await bot.toss(item.type, null, actionObj.quantity || 1);
                return `success: dropped ${actionObj.item}`;
            }
            case "WAIT": {
                const ticks = actionObj.ticks || 20;
                await new Promise(r => setTimeout(r, (ticks/20)*1000));
                return `success: waited ${ticks} ticks`;
            }
            default:
                return `failed: unknown action ${actionObj.action}`;
        }
    } catch (err) {
        return `failed: ${err.message}`;
    }
}

module.exports = { executeAction };
