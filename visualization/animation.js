// ===== CONFIGURATION =====
const ANIMATION_DURATION = 30; // seconds for full replay
const MILESTONE_SLOWDOWN = 0.3; // Speed multiplier during milestones
const MILESTONE_PAUSE = 0.5; // Pause duration at milestone (seconds)

// ===== STATE =====
let data = null;
let animationTimeline = null;
let isPlaying = false;
let currentSpeed = 1;
let currentEpisode = 0;

// ===== CATEGORY COLORS =====
const categoryColors = {
    streak: 'var(--color-streak)',
    mastery: 'var(--color-mastery)',
    unlock: 'var(--color-unlock)',
    performance: 'var(--color-performance)',
    learning: 'var(--color-learning)',
    discovery: 'var(--color-discovery)',
    general: 'var(--text-secondary)'
};

const behaviorColors = {
    sinker: 'var(--color-sinker)',
    dart: 'var(--color-dart)',
    smooth: 'var(--color-smooth)',
    mixed: 'var(--color-mixed)',
    floater: 'var(--color-floater)'
};

// ===== LOAD DATA =====
async function loadData() {
    try {
        const response = await fetch('training_data.json');
        data = await response.json();
        console.log('Data loaded:', data.metadata);
        initializeVisualization();
    } catch (error) {
        console.error('Error loading data:', error);
    }
}

// ===== INITIALIZE VISUALIZATION =====
function initializeVisualization() {
    createTimeline();
    createWinRateChart();
    createBehaviorChart();
    updateStats();
    setupControls();
    createAnimationTimeline();
}

// ===== TIMELINE VISUALIZATION =====
function createTimeline() {
    const svg = d3.select('#timeline-svg');
    const width = svg.node().getBoundingClientRect().width;
    const height = 200;

    svg.attr('width', width).attr('height', height);

    const margin = { top: 40, right: 20, bottom: 40, left: 20 };
    const innerWidth = width - margin.left - margin.right;
    const innerHeight = height - margin.top - margin.bottom;

    const g = svg.append('g')
        .attr('transform', `translate(${margin.left}, ${margin.top})`);

    // X scale for episodes
    const xScale = d3.scaleLinear()
        .domain([0, data.metadata.total_episodes])
        .range([0, innerWidth]);

    // Timeline axis
    const xAxis = d3.axisBottom(xScale)
        .ticks(10)
        .tickFormat(d => `Ep ${d}`);

    g.append('g')
        .attr('transform', `translate(0, ${innerHeight})`)
        .call(xAxis)
        .selectAll('text')
        .attr('class', 'axis-label');

    // Main timeline bar
    g.append('rect')
        .attr('x', 0)
        .attr('y', innerHeight / 2 - 2)
        .attr('width', innerWidth)
        .attr('height', 4)
        .attr('fill', 'var(--border-color)')
        .attr('rx', 2);

    // Progress bar (animated)
    g.append('rect')
        .attr('id', 'timeline-progress')
        .attr('x', 0)
        .attr('y', innerHeight / 2 - 2)
        .attr('width', 0)
        .attr('height', 4)
        .attr('fill', 'var(--accent-primary)')
        .attr('rx', 2);

    // Milestone markers
    data.milestones.forEach(milestone => {
        const x = xScale(milestone.episode);
        const color = categoryColors[milestone.category] || categoryColors.general;

        g.append('line')
            .attr('class', 'milestone-marker')
            .attr('x1', x)
            .attr('x2', x)
            .attr('y1', 0)
            .attr('y2', innerHeight)
            .attr('stroke', color)
            .style('cursor', 'pointer')
            .on('click', () => showMilestonePopup(milestone));

        g.append('circle')
            .attr('cx', x)
            .attr('cy', innerHeight / 2)
            .attr('r', 6)
            .attr('fill', color)
            .style('cursor', 'pointer')
            .on('click', () => showMilestonePopup(milestone));
    });

    // Scrubber (animated position indicator)
    g.append('circle')
        .attr('id', 'timeline-scrubber')
        .attr('cx', 0)
        .attr('cy', innerHeight / 2)
        .attr('r', 10)
        .attr('fill', 'var(--accent-primary)')
        .attr('stroke', 'white')
        .attr('stroke-width', 2);
}

// ===== WIN RATE CHART =====
function createWinRateChart() {
    const svg = d3.select('#winrate-svg');
    const width = svg.node().getBoundingClientRect().width;
    const height = 250;

    svg.attr('width', width).attr('height', height);

    const margin = { top: 20, right: 20, bottom: 40, left: 50 };
    const innerWidth = width - margin.left - margin.right;
    const innerHeight = height - margin.top - margin.bottom;

    const g = svg.append('g')
        .attr('transform', `translate(${margin.left}, ${margin.top})`);

    // Scales
    const xScale = d3.scaleLinear()
        .domain([0, data.episodes.length])
        .range([0, innerWidth]);

    const yScale = d3.scaleLinear()
        .domain([0, 100])
        .range([innerHeight, 0]);

    // Axes
    const xAxis = d3.axisBottom(xScale).ticks(5);
    const yAxis = d3.axisLeft(yScale).ticks(5).tickFormat(d => d + '%');

    g.append('g')
        .attr('transform', `translate(0, ${innerHeight})`)
        .call(xAxis)
        .selectAll('text')
        .attr('class', 'axis-label');

    g.append('g')
        .call(yAxis)
        .selectAll('text')
        .attr('class', 'axis-label');

    // Grid lines
    g.append('g')
        .attr('class', 'grid')
        .call(d3.axisLeft(yScale).ticks(5).tickSize(-innerWidth).tickFormat(''))
        .selectAll('line')
        .attr('class', 'grid-line');

    // Area gradient
    const gradient = svg.append('defs')
        .append('linearGradient')
        .attr('id', 'winrate-gradient')
        .attr('x1', '0%')
        .attr('y1', '0%')
        .attr('x2', '0%')
        .attr('y2', '100%');

    gradient.append('stop')
        .attr('offset', '0%')
        .attr('stop-color', 'var(--accent-primary)')
        .attr('stop-opacity', 0.5);

    gradient.append('stop')
        .attr('offset', '100%')
        .attr('stop-color', 'var(--accent-primary)')
        .attr('stop-opacity', 0);

    // Area
    const area = d3.area()
        .x((d, i) => xScale(i))
        .y0(innerHeight)
        .y1(d => yScale(d.win_rate))
        .curve(d3.curveMonotoneX);

    g.append('path')
        .datum(data.episodes)
        .attr('fill', 'url(#winrate-gradient)')
        .attr('d', area);

    // Line
    const line = d3.line()
        .x((d, i) => xScale(i))
        .y(d => yScale(d.win_rate))
        .curve(d3.curveMonotoneX);

    g.append('path')
        .datum(data.episodes)
        .attr('fill', 'none')
        .attr('stroke', 'var(--accent-primary)')
        .attr('stroke-width', 2)
        .attr('d', line);
}

// ===== BEHAVIOR COMPARISON CHART =====
function createBehaviorChart() {
    const svg = d3.select('#behavior-svg');
    const width = svg.node().getBoundingClientRect().width;
    const height = 250;

    svg.attr('width', width).attr('height', height);

    const margin = { top: 20, right: 80, bottom: 40, left: 50 };
    const innerWidth = width - margin.left - margin.right;
    const innerHeight = height - margin.top - margin.bottom;

    const g = svg.append('g')
        .attr('transform', `translate(${margin.left}, ${margin.top})`);

    // Scales
    const xScale = d3.scaleLinear()
        .domain([0, data.episodes.length])
        .range([0, innerWidth]);

    const yScale = d3.scaleLinear()
        .domain([0, 100])
        .range([innerHeight, 0]);

    // Axes
    const xAxis = d3.axisBottom(xScale).ticks(5);
    const yAxis = d3.axisLeft(yScale).ticks(5).tickFormat(d => d + '%');

    g.append('g')
        .attr('transform', `translate(0, ${innerHeight})`)
        .call(xAxis)
        .selectAll('text')
        .attr('class', 'axis-label');

    g.append('g')
        .call(yAxis)
        .selectAll('text')
        .attr('class', 'axis-label');

    // Grid lines
    g.append('g')
        .attr('class', 'grid')
        .call(d3.axisLeft(yScale).ticks(5).tickSize(-innerWidth).tickFormat(''))
        .selectAll('line')
        .attr('class', 'grid-line');

    // Lines for each behavior
    const behaviors = ['sinker', 'dart', 'smooth', 'mixed', 'floater'];
    const line = d3.line()
        .x((d, i) => xScale(i))
        .y(d => yScale(d))
        .curve(d3.curveMonotoneX);

    behaviors.forEach(behavior => {
        const behaviorData = data.episodes.map(e => e[`${behavior}_rate`]);

        g.append('path')
            .datum(behaviorData)
            .attr('fill', 'none')
            .attr('stroke', behaviorColors[behavior])
            .attr('stroke-width', 2)
            .attr('d', line);

        // Legend
        const legendY = behaviors.indexOf(behavior) * 20;
        g.append('circle')
            .attr('cx', innerWidth + 10)
            .attr('cy', legendY)
            .attr('r', 4)
            .attr('fill', behaviorColors[behavior]);

        g.append('text')
            .attr('x', innerWidth + 20)
            .attr('y', legendY + 4)
            .attr('class', 'axis-label')
            .text(behavior);
    });
}

// ===== UPDATE STATS =====
function updateStats() {
    // Animate counter to final values
    gsap.to({ value: 0 }, {
        value: data.metadata.final_win_rate,
        duration: 1.5,
        ease: 'power2.out',
        onUpdate: function() {
            document.getElementById('final-winrate').textContent = Math.round(this.targets()[0].value) + '%';
        }
    });

    gsap.to({ value: 0 }, {
        value: data.metadata.max_win_streak,
        duration: 1.5,
        ease: 'power2.out',
        onUpdate: function() {
            document.getElementById('max-streak').textContent = Math.round(this.targets()[0].value);
        }
    });

    gsap.to({ value: 0 }, {
        value: data.metadata.total_episodes,
        duration: 1.5,
        ease: 'power2.out',
        onUpdate: function() {
            document.getElementById('total-episodes').textContent = Math.round(this.targets()[0].value);
        }
    });

    gsap.to({ value: 0 }, {
        value: data.metadata.total_milestones,
        duration: 1.5,
        ease: 'power2.out',
        onUpdate: function() {
            document.getElementById('total-milestones').textContent = Math.round(this.targets()[0].value);
        }
    });
}

// ===== MILESTONE POPUP =====
function showMilestonePopup(milestone) {
    const popup = document.getElementById('milestone-popup');
    const category = popup.querySelector('.popup-category');
    const message = popup.querySelector('.popup-message');
    const episode = popup.querySelector('.popup-episode');

    category.textContent = milestone.category;
    category.style.color = categoryColors[milestone.category];
    message.textContent = milestone.message;
    episode.textContent = `Episode ${milestone.episode}`;

    popup.classList.remove('hidden');

    gsap.fromTo(popup,
        { scale: 0.8, opacity: 0 },
        { scale: 1, opacity: 1, duration: 0.3, ease: 'back.out(1.7)' }
    );

    // Auto-hide after 2 seconds
    setTimeout(() => {
        gsap.to(popup, {
            scale: 0.8,
            opacity: 0,
            duration: 0.2,
            onComplete: () => popup.classList.add('hidden')
        });
    }, 2000);
}

// ===== ANIMATION TIMELINE =====
function createAnimationTimeline() {
    const svg = d3.select('#timeline-svg');
    const width = svg.node().getBoundingClientRect().width;
    const margin = { left: 20, right: 20 };
    const innerWidth = width - margin.left - margin.right;

    const xScale = d3.scaleLinear()
        .domain([0, data.metadata.total_episodes])
        .range([0, innerWidth]);

    animationTimeline = gsap.timeline({ paused: true });

    // Animate scrubber and progress bar
    animationTimeline.to('#timeline-scrubber', {
        attr: { cx: innerWidth + margin.left },
        duration: ANIMATION_DURATION,
        ease: 'none',
        onUpdate: function() {
            const progress = this.progress();
            const episodeIndex = Math.floor(progress * data.episodes.length);
            currentEpisode = episodeIndex;

            // Update progress bar
            d3.select('#timeline-progress')
                .attr('width', progress * innerWidth);

            // Check for milestones and trigger popup
            const currentMilestone = data.milestones.find(m => m.episode === data.episodes[episodeIndex]?.episode);
            if (currentMilestone && !currentMilestone.shown) {
                showMilestonePopup(currentMilestone);
                currentMilestone.shown = true;
            }
        }
    }, 0);

    // Add slowdown effect near milestones
    data.milestones.forEach(milestone => {
        const episodeProgress = milestone.episode / data.metadata.total_episodes;
        const timePoint = episodeProgress * ANIMATION_DURATION;

        // Slow down before milestone
        animationTimeline.to({}, {
            duration: 0.5,
            onStart: () => animationTimeline.timeScale(MILESTONE_SLOWDOWN)
        }, timePoint - 0.5);

        // Pause at milestone
        animationTimeline.to({}, {
            duration: MILESTONE_PAUSE
        }, timePoint);

        // Speed back up
        animationTimeline.to({}, {
            duration: 0.3,
            onStart: () => animationTimeline.timeScale(currentSpeed)
        }, timePoint + MILESTONE_PAUSE);
    });
}

// ===== PLAYBACK CONTROLS =====
function setupControls() {
    const playBtn = document.getElementById('play-btn');
    const playIcon = playBtn.querySelector('.play-icon');
    const pauseIcon = playBtn.querySelector('.pause-icon');

    playBtn.addEventListener('click', () => {
        if (isPlaying) {
            animationTimeline.pause();
            playIcon.classList.remove('hidden');
            pauseIcon.classList.add('hidden');
        } else {
            animationTimeline.play();
            playIcon.classList.add('hidden');
            pauseIcon.classList.remove('hidden');
        }
        isPlaying = !isPlaying;
    });

    // Speed controls
    document.querySelectorAll('.speed-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            document.querySelectorAll('.speed-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            currentSpeed = parseFloat(btn.dataset.speed);
            animationTimeline.timeScale(currentSpeed);
        });
    });
}

// ===== INITIALIZE =====
loadData();
