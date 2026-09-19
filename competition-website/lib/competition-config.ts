// Competition configuration
// Edit these values to control competition state

export const COMPETITION_CONFIG = {
    // Competition status: 'upcoming' | 'open' | 'closed'
    status: 'open' as 'upcoming' | 'open' | 'closed',

    // End date for countdown timer (ISO format)
    endDate: '2026-12-31T23:59:59Z',

    // Prize amount
    prizeAmount: 20,

    // YouTube video ID (the part after v= in the URL)
    youtubeVideoId: 'AReU5tDNwKo',

    // GitHub repo URL
    githubRepo: 'https://github.com/keethesh/StardewValleyFishingAI',

    // Published baseline (best checkpoint from the public training run)
    baselineEpisode: 3500,
    baselineModelUrl: '/models/baseline.onnx',
    /** Approximate official-scale score for the published baseline (~eval_score). */
    baselineScore: 0.97,
};
