// This script records only timing information, not the actual keys, for privacy.

const typingArea = document.getElementById('typing-area');
const emotionSelect = document.getElementById('emotion-select');
const submitBtn = document.getElementById('submit-btn');
const resetBtn = document.getElementById('reset-btn');
const messageDiv = document.getElementById('message');

// Stats elements
const avgHoldSpan = document.getElementById('avg-hold');
const avgFlightSpan = document.getElementById('avg-flight');
const totalPausesSpan = document.getElementById('total-pauses');
const avgPauseSpan = document.getElementById('avg-pause');
const wpmSpan = document.getElementById('wpm');

const predictedEmotionP = document.getElementById('predicted-emotion');

// Keystroke timing arrays
let keyEvents = [];      // {downTime, upTime}
let lastKeyUpTime = null;
let startTime = null;
let endTime = null;

// Clear all data
function resetData() {
    keyEvents = [];
    lastKeyUpTime = null;
    startTime = null;
    endTime = null;

    avgHoldSpan.textContent = '-';
    avgFlightSpan.textContent = '-';
    totalPausesSpan.textContent = '-';
    avgPauseSpan.textContent = '-';
    wpmSpan.textContent = '-';
    predictedEmotionP.textContent = '-';
    messageDiv.textContent = '';
    typingArea.value = '';
    emotionSelect.value = '';
}

resetBtn.addEventListener('click', resetData);

// Listen to keydown and keyup on typing area
typingArea.addEventListener('keydown', (event) => {
    const now = performance.now(); // high-resolution time in ms

    if (startTime === null) {
        startTime = now;
    }

    // Start tracking this key event.
    keyEvents.push({
        downTime: now,
        upTime: null,
        flightTime: null // to next key
    });
});

typingArea.addEventListener('keyup', (event) => {
    const now = performance.now();

    if (keyEvents.length === 0) {
        return;
    }

    // Update last key event with upTime
    const current = keyEvents[keyEvents.length - 1];
    current.upTime = now;

    // Compute flight time from previous key up to this key down
    if (lastKeyUpTime !== null) {
        const flight = current.downTime - lastKeyUpTime;
        current.flightTime = flight;
    }

    lastKeyUpTime = now;
    endTime = now;
});

// Compute features on client side to display stats and send raw timing
function computeFeatures() {
    if (keyEvents.length === 0 || startTime === null || endTime === null) {
        return null;
    }

    let holdTimes = [];
    let flightTimes = [];
    let pauseDurations = [];

    for (let i = 0; i < keyEvents.length; i++) {
        const ev = keyEvents[i];
        if (ev.downTime !== null && ev.upTime !== null) {
            const hold = ev.upTime - ev.downTime;
            holdTimes.push(hold);
        }

        if (ev.flightTime !== null) {
            flightTimes.push(ev.flightTime);

            // Count pauses longer than 1000 ms
            if (ev.flightTime > 1000) {
                pauseDurations.push(ev.flightTime);
            }
        }
    }

    const totalDurationMs = endTime - startTime;
    const totalDurationMin = totalDurationMs / 60000.0;

    // Estimate word count using spaces/newlines, purely on client; content is NOT sent to server.
    const text = typingArea.value;
    const words = text.trim().length === 0
        ? 0
        : text.trim().split(/\s+/).length;

    const wpm = totalDurationMin > 0 ? words / totalDurationMin : 0;

    const avgHold = holdTimes.length
        ? holdTimes.reduce((a, b) => a + b, 0) / holdTimes.length
        : 0;

    const avgFlight = flightTimes.length
        ? flightTimes.reduce((a, b) => a + b, 0) / flightTimes.length
        : 0;

    const totalPauses = pauseDurations.length;

    const avgPause = pauseDurations.length
        ? pauseDurations.reduce((a, b) => a + b, 0) / pauseDurations.length
        : 0;

    return {
        avg_key_hold: avgHold,
        avg_flight_time: avgFlight,
        total_pauses: totalPauses,
        avg_pause_duration: avgPause,
        wpm: wpm,
        total_duration_ms: totalDurationMs,
        key_events: keyEvents.map(ev => ({
            downTime: ev.downTime - startTime,
            upTime: ev.upTime !== null ? ev.upTime - startTime : null,
            flightTime: ev.flightTime
        }))
    };
}

function updateStatsDisplay(features) {
    if (!features) return;

    avgHoldSpan.textContent = features.avg_key_hold.toFixed(2);
    avgFlightSpan.textContent = features.avg_flight_time.toFixed(2);
    totalPausesSpan.textContent = features.total_pauses;
    avgPauseSpan.textContent = features.avg_pause_duration.toFixed(2);
    wpmSpan.textContent = features.wpm.toFixed(2);
}

// Submit data to backend
submitBtn.addEventListener('click', async () => {
    const emotion = emotionSelect.value;
    if (!emotion) {
        messageDiv.textContent = 'Please select your current emotion.';
        return;
    }

    const features = computeFeatures();
    if (!features) {
        messageDiv.textContent = 'Please type something first.';
        return;
    }

    updateStatsDisplay(features);

    const payload = {
        timing_data: features.key_events,
        total_duration_ms: features.total_duration_ms,
        wpm: features.wpm,
        emotion_label: emotion
    };

    submitBtn.disabled = true;
    messageDiv.textContent = 'Sending data to server...';

    try {
        const response = await fetch('/api/submit_keystrokes', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(payload)
        });

        if (!response.ok) {
            throw new Error('Server error');
        }

        const data = await response.json();
        if (data.predicted_emotion) {
            predictedEmotionP.textContent = data.predicted_emotion;
            messageDiv.textContent = 'Data saved and emotion predicted.';
        } else {
            messageDiv.textContent = 'Data saved, but prediction not available yet (need more training data).';
        }
    } catch (err) {
        console.error(err);
        messageDiv.textContent = 'Failed to send data to server.';
    } finally {
        submitBtn.disabled = false;
    }
});
