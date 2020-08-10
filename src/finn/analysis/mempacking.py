# Copyright (c) 2020, Xilinx
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of FINN nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# Implements memory packing as described by https://arxiv.org/abs/2003.12449
# to determine the best way to combine weight buffers in on-chip memory (OCM) to
# achieve maximum OCM utilization efficiency, or conversely the minimum number
# of BRAMs required to store the NN weights.
#
# Input: memory shapes as provided by the shape analysis pass
# Output: a dictionary of numbered bins each of which contains one or more PE
# weight buffers.
#
# Example output for a neural net of 2 layers (0 and 1), of which layer0 has
# 4 PEs and layer1 has 2 PEs:
# {'bin0': ['layer0', 'layer0', 'layer1'], 'bin1': ['layer0', 'layer0', 'layer1']}
# this indicates that the weight memories of the two layers are best
# assembled into two contiguous RAMs corresponding to each of the two bins in
# the dictionary, and each bin/RAM contains two PEs from layer0 and one from layer1
# Note: it is not relevant which PEs specifically go into each bin, as they are
# of identical shape

import random
import math

from deap import base
from deap import creator
from deap import tools

from finn.analysis.shapes import weight_shapes


# initialize chromosomes with cardinality constraints
def cardinalInit(args):
    new_ind = []
    inv = list(range(args.start_id, args.start_id + args.end_id))
    # apply "intra-layer" stacking on all partitions
    if random.random() < args.gold:
        new_ind = simpleStack(inv, args)
    # shuffle the partitions around with probability: 1 - gold
    else:
        new_ind = simpleStack(random.sample(inv, args.end_id), args)
    return new_ind


# stack adjacent memory partitions up to max_stack height
def simpleStack(inventory, args):
    stacks = []
    cur_stack = []
    h = 0
    cur_layer = args.lookupT[inventory[0]][0]
    while len(list(inventory)) > 0:
        new_layer = args.lookupT[inventory[0]][0]
        if (h < args.max_stack_height) and (
            (cur_layer == new_layer) or args.enableInter
        ):
            cur_stack.append(inventory[0])
            inventory.remove(inventory[0])
            h += 1
        else:
            stacks.append(cur_stack)
            cur_stack = []
            h = 0
            cur_layer = args.lookupT[inventory[0]][0]
    if len(cur_stack) > 0:
        stacks.append(cur_stack)
    return stacks


# first fit heuristic with dynamic bin capacity
def firstFitDynamic(inventory, args):
    stacks = []
    cur_stack = []
    h = 0
    w = 0
    overshoot = 0
    new_overshoot = 0
    pow2_overshoot = 0
    new_pow2_overshoot = 0
    stack_h = 0
    cur_layer = args.lookupT[inventory[0]][0]
    while len(inventory) > 0:
        for item in list(inventory):
            new_layer = args.lookupT[item][0]

            if stack_h == 0:
                w = args.lookupT[item][1] * args.lookupT[item][4]
            # new bin height if the item were to be packed in the same stack
            new_h = h + args.lookupT[item][3]
            # adjust the height of the memory resource based on the BRAM "aspect" mode
            (dyn_mem_height, k) = aspectRatio(w, args)
            # readjust the height for the case that the current item will join the stack
            (dyn_mem_height_new, k) = aspectRatio(
                max(w, args.lookupT[item][1] * args.lookupT[item][4]), args
            )
            # compute how well the current stack fits in the memory resource
            if args.aloc_pow2:
                if h > 0:
                    pow2_overshoot = max(
                        abs(h - 2 ** math.ceil(math.log(h) / math.log(2))),
                        abs((h % dyn_mem_height) - dyn_mem_height),
                    )
                else:
                    pow2_overshoot = dyn_mem_height

            overshoot = abs((h % dyn_mem_height) - dyn_mem_height)
            if h == dyn_mem_height:
                overshoot = 0
            # and how well the stack fits after adding the item
            if args.aloc_pow2:
                if new_h > 0:
                    new_pow2_overshoot = max(
                        abs(new_h - 2 ** math.ceil(math.log(new_h) / math.log(2))),
                        abs((new_h % dyn_mem_height_new) - dyn_mem_height_new),
                    )
                else:
                    new_pow2_overshoot = dyn_mem_height
            new_overshoot = abs((new_h % dyn_mem_height_new) - dyn_mem_height_new)
            if new_h == dyn_mem_height_new:
                new_overshoot = 0
            # only add the item if it is the first in the bin or makes
            # the stack fit better in the memory resource
            if (stack_h == 0) or (
                (
                    (stack_h < args.max_stack_height)
                    and (cur_layer == new_layer or args.enableInter)
                )
                and (
                    w == (args.lookupT[item][1] * args.lookupT[item][4])
                    or (random.random() <= args.p_adm)
                )
                and (
                    ((random.random() <= args.mut_gen) or (new_overshoot < overshoot))
                    or (
                        args.aloc_pow2
                        and (new_pow2_overshoot < pow2_overshoot)
                        and overshoot == new_overshoot
                    )
                )
            ):
                cur_stack.append(item)
                inventory.remove(item)
                stack_h += 1
                h += args.lookupT[item][3]
                w = max(w, args.lookupT[item][1] * args.lookupT[item][4])
            # finalize the current stack and place the item in a new bin otherwise
            else:
                stacks.append(cur_stack)
                cur_stack = []
                cur_stack.append(item)
                inventory.remove(item)
                stack_h = 1
                h = args.lookupT[item][3]
                w = args.lookupT[item][1] * args.lookupT[item][4]
                cur_layer = args.lookupT[item][0]
    # finalize any leftover stacks
    if len(cur_stack) > 0:
        stacks.append(cur_stack)
    return stacks


# adapt memory resource height based on the BRAM mode
def aspectRatio(partition_w, args):
    dyn_height = 0
    dyn_width = 0
    mem_height = 1024
    mem_width = 16
    if args.primitive == "bram":
        if partition_w <= 1:  # case "narrow" 1-bit x 16384 aspect ratio cost
            dyn_height = 16 * mem_height
            dyn_width = mem_width / 16
        elif partition_w <= 2:  # case "narrow" 2-bit x 8192 aspect ratio cost
            dyn_height = 8 * mem_height
            dyn_width = mem_width / 8
        elif partition_w <= 4:  # case "narrow" 4-bit x 4096 aspect ratio cost
            dyn_height = 4 * mem_height
            dyn_width = mem_width / 4
        elif partition_w <= 8:  # case "narrow" 8-bit x 2048 aspect ratio cost
            dyn_height = 2 * mem_height
            dyn_width = mem_width / 2
        else:  # regular BRAM cost (i.e. 16-bit x 1024)
            dyn_height = mem_height
            dyn_width = mem_width
    else:
        dyn_height = mem_height
        dyn_width = mem_width

    return dyn_height, dyn_width


def mapEff(individual, args):
    imap = []
    for bin in individual:
        h = 0  # bin height
        w = 0  # bin width
        mem_cap = 0  # actual data stored in the bin [bits]
        for part in bin:
            item_height = args.lookupT[part][3]
            item_width = args.lookupT[part][1] * args.lookupT[part][4]
            mem_cap += item_width * item_height
            h += item_height
            w = max(item_width, w)
        dyn_mem_height, dyn_mem_width = aspectRatio(w, args)
        if args.aloc_pow2:
            bram_count = math.ceil(
                2 ** math.ceil(math.log(h) / math.log(2)) / dyn_mem_height
            ) * math.ceil(w / dyn_mem_width)
        else:
            bram_count = math.ceil(h / dyn_mem_height) * math.ceil(w / dyn_mem_width)
        map_efficiency = mem_cap / (bram_count * 16 * 1024)
        imap.append((bin, map_efficiency))

    return imap


def getKey(item):
    return item[1]


def mutateValid(args, individual):
    # new_ind = individual
    inventory = []
    repacked = []
    # new_ind2 = random.sample(new_ind, len(new_ind))
    new_ind = sorted(mapEff(individual, args), key=getKey)
    new_ind2 = []
    ctr = 0
    for item in new_ind:
        if item[1] < 1.0:
            ctr += 1
        new_ind2.append(item[0])
    # rng = min(0.2*len(new_ind2), ctr)
    for reps in range(
        0, ctr
    ):  # TODO: make random and adapt rng to range(1, len(individual))
        # target = random.randint(0, ctr)
        for gene in list(new_ind2[0]):
            inventory.append(gene)
        new_ind2.remove(new_ind2[0])
    if random.random() <= 0.5:
        repacked = firstFitDynamic(random.sample(inventory, len(inventory)), args)
    else:
        repacked = firstFitDynamic(inventory, args)
    for new_bin in repacked:
        new_ind2.append(new_bin)
    return new_ind2


def cxOver(args, solution, solution2):
    perturbed = []
    for col in solution:
        perturbed.append(col.copy())

    ind1 = random.randint(int(len(solution2) / 2), int(len(solution2)) - 1)
    ind2 = random.randint(int(len(solution2) / 2), int(len(solution2)) - 1)

    while ind1 == ind2:
        ind1 = random.randint(0, len(perturbed) - 1)
        ind2 = random.randint(0, len(perturbed) - 1)

    lo = min(ind1, ind2)
    hi = max(ind1, ind2)

    snippet = solution2[lo:hi].copy()
    inventory_raw = []
    for candidate in snippet:
        for item in candidate:
            ctr = 0
            for bin in list(perturbed):
                if item in list(bin):
                    cbin = bin.copy()
                    cbin.remove(item)
                    if len(list(cbin)) > 0:
                        inventory_raw.append(cbin)

                    perturbed.pop(ctr)
                    break
                ctr += 1
        perturbed.append(candidate)

    inventory = []
    if len(inventory_raw) > 0:
        for lst in inventory_raw:
            for item in lst:
                inventory.append(item)
        repacked = firstFitDynamic(random.sample(inventory, len(inventory)), args)
        for bin in repacked:
            perturbed.append(bin)

    return perturbed


# Fitness evaluation function (decreases BRAM cost and balances throughput for now)
def getFitness(args, individual):
    bin_metrics = []
    mem_height = 1024
    mem_width = 16
    for bin in individual:
        hi_stack = 0
        h = 0  # bin height
        w = 0  # bin width
        mem_cap = 0  # actual data stored in the bin [bits]
        bram_count = 0  # RAMB18 modules spanned by the bin
        stack_height = 0  # amount of depth-wise stacked partitions within the bin
        clayers = []
        for part in bin:
            # Keep track of the capacity stored in the bin
            clayers.append(args.lookupT[part][0])
            item_height = args.lookupT[part][3]
            item_width = args.lookupT[part][1] * args.lookupT[part][4]
            mem_cap += item_width * item_height
            # and update the dimensions of the bin
            h += item_height
            w = max(item_width, w)
            stack_height += 1
        # Compute the BRAM span of the bin
        dyn_mem_height, dyn_mem_width = aspectRatio(w, args)
        if args.aloc_pow2:
            bram_count = math.ceil(
                2 ** math.ceil(math.log(h) / math.log(2)) / dyn_mem_height
            ) * math.ceil(w / dyn_mem_width)
        else:
            bram_count = math.ceil(h / dyn_mem_height) * math.ceil(w / dyn_mem_width)
        if stack_height == 1:
            if h <= int(mem_height / 2):
                bram_count = math.ceil((2 * h) / dyn_mem_height) * math.ceil(
                    w / (2 * mem_width)
                )
        if stack_height > 2:
            hi_stack = 1
        layer_count = len(set(clayers))
        map_efficiency = mem_cap / (bram_count * mem_width * mem_height)
        bin_metrics.append(
            (bin, w, h, bram_count, map_efficiency, stack_height, layer_count, hi_stack)
        )

    # Fitness values to optimize for (total BRAM cost, max stack height)
    bram = sum([metric[3] for metric in bin_metrics])
    stack = sum([metric[5] for metric in bin_metrics]) / len(individual)
    comp = sum([metric[6] for metric in bin_metrics])
    lut_complexity = sum([metric[7] for metric in bin_metrics])

    return (
        bram + comp / 1000 + 10 * (stack / args.max_stack_height),
        bram,
        lut_complexity,
    )


class packDefaultConfig:
    thresh_min = 129  # Minimum depth at which memory is mapped to BRAM
    thresh_max = 2 ** 20  # Maximum depth at which memory is mapped to BRAM

    enableInter = True  # Do inter-layer packing
    aloc_pow2 = False  # Constrain stacks to power-of-2 depths
    max_stack_height = 4  # Maximum height of a stack
    primitive = "bram"  # specify memory resource primitive', required=False)

    # GA parameters
    t_con = 100  # proportionality constant for the latency (fitness function)
    mut = 0.3  # mutation probability
    mut_gen = 0.001  # probability of mutating a gene in chromosome
    p_adm = 0.1  # probability of admitting memories with mismatching widths
    cross = 0.0  # crossover probability
    gold = 0.125  # probability of "golden" initialization (pre-packed intra-layer)
    top = 5  # top-x fittest selection size
    pop_count = 100  # Size of the population per iteration in the GA
    generations = 100  # Generation limit after which the GA is stopped


def pack_memory(model, args=packDefaultConfig()):
    # scrape weight shapes from model
    shapes = weight_shapes(model)
    # do packing
    return pack_memory_shapes(shapes, args)


def pack_memory_shapes(shapes, args=packDefaultConfig()):
    opt = (-1.0, -1.0, -1.0)  # optimization weights
    args.start_id = 0
    line_count = 0
    layers = 0
    partitions = 0
    args.lookupT = []
    mapped_layers = []

    # network parser
    for layer in shapes:
        label = layer
        simd = shapes[layer]["SIMD"]
        pe = shapes[layer]["PE"]
        wmem = shapes[layer]["WMEM"]
        wprec = shapes[layer]["DataType"].bitwidth()
        # Only record layers that map reasonably efficiently to BRAM
        if wmem >= args.thresh_min and wmem <= args.thresh_max:
            mapped_layers.append(layers)
            for partition in range(0, pe):
                # Look-up table to search for memory partition dimensions
                args.lookupT.append((layers, simd, pe, wmem, wprec, label))
                partitions += 1
        line_count += 1
        layers += 1

    assert (
        len(mapped_layers) != 0
    ), "No suitable layers to process. Try reducing your threshold"

    args.end_id = partitions

    # Create individual and fitness parameter
    creator.create("MultiModeFit", base.Fitness, weights=opt)
    creator.create("Individual", list, fitness=creator.MultiModeFit)

    # Register the population related tools to the toolbox
    toolbox = base.Toolbox()
    toolbox.register("initGenes", cardinalInit, args)
    toolbox.register(
        "individual", tools.initIterate, creator.Individual, toolbox.initGenes
    )
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    # Register the Genetic Operators to the toolbox
    toolbox.register("evaluate", getFitness, args)
    toolbox.register("mate", cxOver, args)
    toolbox.register("mutate", mutateValid, args)
    toolbox.register("select", tools.selTournament, tournsize=args.top)

    result = report(optimize(toolbox, args), args)
    return result


def optimize(toolbox, args):
    pop = toolbox.population(n=args.pop_count)

    # Evaluate the entire population
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, (bram_c, stack_h, lut_c) in zip(pop, fitnesses):
        ind.fitness.values = (bram_c, stack_h, lut_c)

    g = 0
    while g < args.generations:
        # Generate the new generation
        if args.max_stack_height == 1:
            # if all bins have height == 1, stop
            g = g + args.generations
        else:
            g = g + 1

        offspring = toolbox.select(pop, len(pop))
        offspring = list(map(toolbox.clone, offspring))

        # Perform mutation
        ctr = 0
        for mutant in offspring:
            if random.random() < args.mut:
                offspring[ctr][:] = toolbox.mutate(mutant)
                del mutant.fitness.values
            ctr += 1

        # Update the fitness values of the mutants and offspring
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, (fit0, fit1, fit2) in zip(invalid_ind, fitnesses):
            ind.fitness.values = (fit0, fit1, fit2)

        # Repeat the cycle of life
        pop[:] = offspring
        compound_fitness = [ind.fitness.values[0] for ind in pop]

        # Track the chromosome of the fittest individual (in terms of BRAM count)
        fittest = [
            indiv for indiv in pop if indiv.fitness.values[0] == min(compound_fitness)
        ][0]

    return fittest


def report(fittest, args):
    count = 0
    result = {}
    for cbin in fittest:
        result["bin" + str(count)] = [args.lookupT[i + args.start_id][5] for i in cbin]
        count += 1
    return result
