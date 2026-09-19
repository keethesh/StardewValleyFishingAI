import { createClient, RedisClientType } from 'redis';

let redis: RedisClientType | null = null;

export async function getRedis() {
    if (!redis) {
        redis = createClient({ url: process.env.KV_URL });
        await redis.connect();
    }
    return redis;
}
