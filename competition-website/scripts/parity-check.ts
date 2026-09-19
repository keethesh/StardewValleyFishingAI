/**
 * Compare TypeScript FishingGame against a Python-generated fixture.
 * Run via: npx tsx scripts/parity-check.ts [fixture.json]
 */
import * as fs from 'fs';
import * as path from 'path';
import { FishingGame, Fish } from '../lib/game-logic';
import { PortableRNG } from '../lib/portable-rng';

const TOL = 1e-9;

function assertClose(label: string, a: number, b: number, tol = TOL) {
    if (Math.abs(a - b) > tol) {
        throw new Error(`${label}: TS=${a} Python=${b} (Δ=${Math.abs(a - b)})`);
    }
}

function rngSelfTest() {
    const rng = new PortableRNG(42);
    const vals = Array.from({ length: 5 }, () => rng.nextUint32());
    const expected = [2581720956, 1925393290, 3661312704, 2876485805, 750819978];
    for (let i = 0; i < expected.length; i++) {
        if (vals[i] !== expected[i]) {
            throw new Error(`PortableRNG golden mismatch at ${i}: ${vals[i]} !== ${expected[i]}`);
        }
    }
    console.log('PortableRNG self-test OK');
}

function main() {
    rngSelfTest();

    const fixturePath =
        process.argv[2] || path.join(__dirname, 'parity-fixture.json');
    const fixture = JSON.parse(fs.readFileSync(fixturePath, 'utf8'));

    const fish: Fish = fixture.fish;
    const game = new FishingGame(fish, fixture.seed, { augmentFish: false });
    game.reset(fixture.seed);

    const f0 = fixture.frames[0];
    assertClose('t0.bobberPosition', game.state.bobberPosition, f0.bobberPosition);
    assertClose('t0.bobberBarPos', game.state.bobberBarPos, f0.bobberBarPos);
    for (let i = 0; i < f0.obs.length; i++) {
        assertClose(`t0.obs[${i}]`, game.getObservation()[i], f0.obs[i]);
    }

    let mismatches = 0;
    for (let i = 0; i < fixture.actions.length; i++) {
        const action = fixture.actions[i] as 0 | 1;
        const result = game.step(action);
        const expected = fixture.frames[i + 1];

        const checks: [string, number, number][] = [
            ['bobberPosition', game.state.bobberPosition, expected.bobberPosition],
            ['bobberSpeed', game.state.bobberSpeed, expected.bobberSpeed],
            ['bobberTargetPosition', game.state.bobberTargetPosition, expected.bobberTargetPosition],
            ['bobberBarPos', game.state.bobberBarPos, expected.bobberBarPos],
            ['bobberBarSpeed', game.state.bobberBarSpeed, expected.bobberBarSpeed],
            [
                'floaterSinkerAcceleration',
                game.state.floaterSinkerAcceleration,
                expected.floaterSinkerAcceleration,
            ],
            ['distanceFromCatching', game.state.distanceFromCatching, expected.distanceFromCatching],
            ['reward', result.reward, expected.reward],
        ];

        for (const [label, a, b] of checks) {
            try {
                assertClose(`t${i + 1}.${label}`, a, b);
            } catch (e) {
                console.error(String(e));
                mismatches++;
            }
        }

        if (Boolean(game.state.bobberInBar) !== Boolean(expected.bobberInBar)) {
            console.error(
                `t${i + 1}.bobberInBar: TS=${game.state.bobberInBar} Python=${expected.bobberInBar}`
            );
            mismatches++;
        }

        for (let j = 0; j < expected.obs.length; j++) {
            try {
                assertClose(`t${i + 1}.obs[${j}]`, result.obs[j], expected.obs[j]);
            } catch (e) {
                console.error(String(e));
                mismatches++;
            }
        }

        if (mismatches > 20) {
            console.error('Too many mismatches — aborting');
            process.exit(1);
        }

        if (result.done) break;
    }

    if (mismatches > 0) {
        console.error(`FAIL: ${mismatches} mismatches`);
        process.exit(1);
    }
    console.log(`PASS: ${fixture.actions.length} steps match Python fixture`);
}

main();
