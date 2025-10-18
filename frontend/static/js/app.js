const API_BASE = '';

const LOCATION_LABELS = [
    { name: 'Red', coords: '(0, 0)' },
    { name: 'Green', coords: '(0, 4)' },
    { name: 'Yellow', coords: '(4, 0)' },
    { name: 'Blue', coords: '(4, 3)' },
];

document.addEventListener('DOMContentLoaded', () => {
    const resetButton = document.getElementById('reset-button');
    const stepButton = document.getElementById('step-button');
    const autoStepButton = document.getElementById('auto-step-button');

    const statusIndicator = document.getElementById('status-indicator');
    const gridContainer = document.getElementById('taxi-grid');
    const stepsValue = document.getElementById('steps-value');
    const rewardValue = document.getElementById('reward-value');
    const stateDescription = document.getElementById('state-description');
    const taxiCoords = document.getElementById('taxi-coords');
    const passengerLocation = document.getElementById('passenger-location');
    const destinationLocation = document.getElementById('destination-location');
    const llmThinking = document.getElementById('llm-thinking');
    const llmAction = document.getElementById('llm-action');
    const historyList = document.getElementById('history-list');

    let autoInterval = null;
    let previousReward = 0;
    let history = [];

    resetButton.addEventListener('click', async () => {
        await postAction('reset', { recordHistory: false });
        history = [];
        previousReward = 0;
        refreshHistory();
        stopAutoPlay();
    });

    stepButton.addEventListener('click', () => postAction('step', { recordHistory: true }));

    autoStepButton.addEventListener('click', () => {
        if (autoInterval) {
            stopAutoPlay();
        } else {
            startAutoPlay();
        }
    });

    async function postAction(endpoint, { recordHistory }) {
        disableControls(true);
        try {
            if (endpoint === 'step') {
                await streamStep(recordHistory);
            } else {
                const response = await fetch(`${API_BASE}/${endpoint}`, { method: 'POST' });
                const data = await response.json();
                handleResponse(data, { recordHistory });
            }
        } catch (error) {
            console.error(`Error calling ${endpoint}:`, error);
            setStatus('error', 'Network error');
        } finally {
            disableControls(false);
        }
    }

    async function streamStep(recordHistory) {
        const response = await fetch(`${API_BASE}/step`, { method: 'POST' });

        if (!response.ok) {
            throw new Error(`Step request failed with status ${response.status}`);
        }

        if (!response.body) {
            throw new Error('Streaming responses are not supported in this browser.');
        }

        setStatus('active', 'LLM thinking…');
        llmThinking.textContent = '';

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';
        let accumulatedThinking = '';

        const applyFinalThinking = (payload) => {
            if (!payload) return;
            const preferred =
                payload.llm_thinking && payload.llm_thinking.length > accumulatedThinking.length
                    ? payload.llm_thinking
                    : accumulatedThinking || payload.raw_response || '';
            if (preferred) {
                llmThinking.textContent = preferred;
            }
        };

        while (true) {
            const { value, done } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });

            let newlineIndex = buffer.indexOf('\n');
            while (newlineIndex !== -1) {
                const rawLine = buffer.slice(0, newlineIndex).trim();
                buffer = buffer.slice(newlineIndex + 1);

                if (rawLine) {
                    let event;
                    try {
                        event = JSON.parse(rawLine);
                    } catch (parseError) {
                        console.error('Failed to parse stream chunk:', parseError, rawLine);
                        event = null;
                    }

                    if (event) {
                        if (event.type === 'token') {
                            accumulatedThinking += event.token || '';
                            llmThinking.textContent = accumulatedThinking;
                        } else if (event.type === 'error') {
                            setStatus('error', event.error || 'Model error');
                            stopAutoPlay();
                            return;
                        } else if (event.type === 'final') {
                            if (event.payload) {
                                handleResponse(event.payload, { recordHistory });
                                applyFinalThinking(event.payload);
                            }
                            return;
                        }
                    }
                }

                newlineIndex = buffer.indexOf('\n');
            }
        }

        buffer += decoder.decode();

        let newlineIndex = buffer.indexOf('\n');
        while (newlineIndex !== -1) {
            const rawLine = buffer.slice(0, newlineIndex).trim();
            buffer = buffer.slice(newlineIndex + 1);

            if (rawLine) {
                try {
                    const event = JSON.parse(rawLine);
                    if (event.type === 'token') {
                        accumulatedThinking += event.token || '';
                        llmThinking.textContent = accumulatedThinking;
                    } else if (event.type === 'error') {
                        setStatus('error', event.error || 'Model error');
                        stopAutoPlay();
                        return;
                    } else if (event.type === 'final') {
                        if (event.payload) {
                            handleResponse(event.payload, { recordHistory });
                            applyFinalThinking(event.payload);
                        }
                        return;
                    }
                } catch (parseError) {
                    console.error('Failed to parse trailing stream chunk:', parseError, rawLine);
                }
            }

            newlineIndex = buffer.indexOf('\n');
        }

        setStatus('error', 'Streaming was interrupted unexpectedly.');
        stopAutoPlay();
    }

    async function fetchState() {
        try {
            const response = await fetch(`${API_BASE}/state`);
            const data = await response.json();
            handleResponse(data, { recordHistory: false });
            previousReward = data.total_reward ?? 0;
        } catch (error) {
            console.error('Error fetching state:', error);
            setStatus('error', 'Unable to reach backend');
        }
    }

    function handleResponse(data, { recordHistory }) {
        if (data.error) {
            alert(data.error);
            setStatus('error', data.error);
            return;
        }

        updateDisplay(data);

        if (data.llm_error) {
            setStatus('error', data.llm_error);
            stopAutoPlay();
            return;
        }

        if (recordHistory) {
            const deltaReward = (data.total_reward ?? 0) - previousReward;
            previousReward = data.total_reward ?? 0;
            addHistoryEntry(data, deltaReward);
        }

        if (data.is_over) {
            setStatus('complete', 'Episode finished');
            stopAutoPlay();
        } else if ((data.steps ?? 0) === 0) {
            setStatus('idle', 'Ready');
        } else {
            setStatus('active', 'Running');
        }
    }

    function updateDisplay(data) {
        stepsValue.textContent = `Steps: ${data.steps ?? 0}`;
        rewardValue.textContent = `Reward: ${(data.total_reward ?? 0).toFixed(2)}`;
        stateDescription.textContent = data.state_description || '—';

        if (data.state) {
            taxiCoords.textContent = `(${data.state.taxi_row}, ${data.state.taxi_col})`;

            if (data.state.passenger_location === 4) {
                passengerLocation.textContent = 'Inside taxi';
            } else {
                const info = LOCATION_LABELS[data.state.passenger_location];
                passengerLocation.textContent = `${info.name} ${info.coords}`;
            }

            const destInfo = LOCATION_LABELS[data.state.destination_index];
            destinationLocation.textContent = `${destInfo.name} ${destInfo.coords}`;
        }

        llmThinking.textContent = data.llm_thinking || '—';
        const actionText = data.llm_action_text ? `${data.llm_action_text} (${data.llm_action_code})` : '–';
        llmAction.textContent = `Action: ${actionText}`;

        renderGrid(data.grid);
    }

    function addHistoryEntry(data, rewardDelta) {
        if (data.llm_error) return;

        const entry = {
            step: data.steps,
                action:
                    data.llm_action_text ||
                    (data.llm_action_code !== undefined && data.llm_action_code !== null
                        ? `Action ${data.llm_action_code}`
                        : 'No action'),
            reward: rewardDelta,
            thinking: data.llm_thinking,
        };

        history = [entry, ...history].slice(0, 12);
        refreshHistory();
    }

    function refreshHistory() {
        historyList.innerHTML = '';

        if (!history.length) {
            const empty = document.createElement('li');
            empty.className = 'history-empty';
            empty.textContent = 'Steps will appear here once the agent acts.';
            historyList.appendChild(empty);
            return;
        }

        history.forEach((item) => {
            const li = document.createElement('li');
            const header = document.createElement('div');
            header.className = 'history-row';

            const stepLabel = document.createElement('strong');
            stepLabel.textContent = `Step ${item.step}`;
            header.appendChild(stepLabel);

            const actionSpan = document.createElement('span');
            actionSpan.className = 'history-action';
            actionSpan.textContent = ` — ${item.action || 'unknown action'}`;
            header.appendChild(actionSpan);

            const rewardBadge = rewardBadgeFor(item.reward);
            if (rewardBadge) {
                header.appendChild(rewardBadge);
            }

            const thinking = document.createElement('p');
            thinking.className = 'history-thinking';
            thinking.textContent = item.thinking || '';

            li.appendChild(header);
            li.appendChild(thinking);
            historyList.appendChild(li);
        });
    }

    function rewardBadgeFor(value) {
        if (value === undefined || Number.isNaN(value)) return null;
        const formatted = value > 0 ? `+${value.toFixed(1)}` : value.toFixed(1);
        const tone = value > 0 ? 'success' : value < 0 ? 'danger' : 'neutral';
        const badge = document.createElement('span');
        badge.className = `reward-badge ${tone}`;
        badge.textContent = formatted;
        return badge;
    }

    function createGridPlaceholder(message) {
        gridContainer.innerHTML = '';
        const placeholder = document.createElement('span');
        placeholder.className = 'grid-placeholder';
        placeholder.textContent = message;
        gridContainer.appendChild(placeholder);
    }

    function renderGrid(grid) {
        if (!grid || !Array.isArray(grid.cells) || !grid.cells.length) {
            gridContainer.classList.add('is-empty');
            createGridPlaceholder('Press reset to render the board.');
            return;
        }

        gridContainer.classList.remove('is-empty');
        gridContainer.innerHTML = '';
        const cols = grid.cols ?? (grid.cells[0] ? grid.cells[0].length : 0);
        const rows = grid.rows ?? grid.cells.length;
        gridContainer.style.setProperty('--grid-cols', cols);
        gridContainer.style.setProperty('--grid-rows', rows);
    gridContainer.dataset.passenger = grid.passenger_in_taxi ? 'onboard' : 'waiting';

        grid.cells.forEach((row) => {
            row.forEach((cell) => {
                const cellDiv = document.createElement('div');
                cellDiv.classList.add('grid-cell');
                cellDiv.dataset.row = cell.row;
                cellDiv.dataset.col = cell.col;

                if (cell.walls) {
                    Object.entries(cell.walls).forEach(([direction, hasWall]) => {
                        if (hasWall) {
                            cellDiv.classList.add(`wall-${direction}`);
                        }
                    });
                }

                if (cell.is_taxi) {
                    cellDiv.classList.add('has-taxi');
                }
                if (cell.is_destination) {
                    cellDiv.classList.add('is-destination');
                }
                if (cell.has_passenger) {
                    cellDiv.classList.add('has-passenger');
                }

                if (cell.landmark) {
                    const landmark = document.createElement('span');
                    landmark.className = `cell-landmark landmark-${cell.landmark.code.toLowerCase()}`;
                    landmark.textContent = cell.landmark.code;
                    landmark.title = `${cell.landmark.name} stop`;
                    cellDiv.appendChild(landmark);
                }

                if (cell.is_destination) {
                    const destination = document.createElement('span');
                    destination.className = 'cell-marker destination';
                    destination.textContent = '★';
                    destination.title = grid.destination_code
                        ? `Destination (${grid.destination_code})`
                        : 'Destination';
                    cellDiv.appendChild(destination);
                }

                const actors = document.createElement('div');
                actors.className = 'cell-actors';

                if (cell.has_passenger && !grid.passenger_in_taxi) {
                    const passenger = document.createElement('span');
                    passenger.className = 'cell-icon passenger';
                    passenger.textContent = '🧍';
                    passenger.title = 'Passenger waiting';
                    actors.appendChild(passenger);
                }

                if (cell.is_taxi) {
                    const taxi = document.createElement('span');
                    taxi.className = 'cell-icon taxi';
                    taxi.textContent = '🚕';
                    taxi.title = grid.passenger_in_taxi ? 'Taxi — passenger onboard' : 'Taxi';
                    actors.appendChild(taxi);

                    if (grid.passenger_in_taxi) {
                        const onboard = document.createElement('span');
                        onboard.className = 'cell-badge onboard';
                        onboard.textContent = '🧍';
                        onboard.title = 'Passenger onboard';
                        cellDiv.appendChild(onboard);
                    }
                }

                if (actors.children.length) {
                    cellDiv.appendChild(actors);
                }

                gridContainer.appendChild(cellDiv);
            });
        });
    }

    function setStatus(mode, label) {
        statusIndicator.textContent = label;
        statusIndicator.classList.remove('complete', 'error');
        if (mode === 'complete') {
            statusIndicator.classList.add('complete');
        } else if (mode === 'error') {
            statusIndicator.classList.add('error');
        }
    }

    function disableControls(flag) {
        resetButton.disabled = flag;
        stepButton.disabled = flag;
        autoStepButton.disabled = flag;
    }

    function startAutoPlay() {
        setStatus('active', 'Auto-play');
        autoStepButton.textContent = 'Stop auto-play';
        autoInterval = setInterval(async () => {
            if (!autoInterval) return;
            await postAction('step', { recordHistory: true });
        }, 2500);
    }

    function stopAutoPlay() {
        if (autoInterval) {
            clearInterval(autoInterval);
            autoInterval = null;
        }
        autoStepButton.textContent = 'Start auto-play';
    }

    // Initial load
    fetchState();
});