class MultiAssetDashboard {
    constructor() {
        this.baseUrl = '';
        this.updateInterval = 3000; // 3 seconds
        this.updateTimer = null;
        this.assets = ['crypto', 'futures', 'forex', 'forex_options'];
        this.charts = {};
        
        // Auto-reconnect system
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 15;
        this.isConnected = true;
        this.reconnectDelay = 5000; // 5 seconds between reconnect attempts
        this.lastSuccessfulUpdate = Date.now();
        
        this.init();
    }

    init() {
        this.bindEvents();
        this.initCharts();
        this.loadInitialData();
        this.startPeriodicUpdates();
        this.logActivity('System', 'Dashboard initialized');
    }

    bindEvents() {
        // Emergency stop button
        const emergencyBtn = document.getElementById('emergencyStop');
        if (emergencyBtn) {
            emergencyBtn.addEventListener('click', () => this.emergencyStopAll());
        }
    }

    initCharts() {
        this.assets.forEach(asset => {
            // Initialize paper trading chart
            const paperCanvas = document.getElementById(`${asset}-paper-chart`);
            if (paperCanvas) {
                this.charts[`${asset}-paper`] = new Chart(paperCanvas, {
                    type: 'line',
                    data: {
                        labels: [],
                        datasets: [{
                            label: 'Portfolio Balance',
                            data: [],
                            borderColor: '#10b981',
                            backgroundColor: 'rgba(16, 185, 129, 0.15)',
                            tension: 0.3,
                            fill: true,
                            borderWidth: 2,
                            pointRadius: 0,
                            pointHoverRadius: 4,
                            pointHoverBorderWidth: 2,
                            pointHoverBackgroundColor: '#10b981',
                            pointHoverBorderColor: '#ffffff'
                        }]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        interaction: {
                            intersect: false,
                            mode: 'index'
                        },
                        plugins: {
                            legend: { display: false },
                            tooltip: {
                                backgroundColor: 'rgba(26, 31, 46, 0.95)',
                                titleColor: '#ffffff',
                                bodyColor: '#94a3b8',
                                borderColor: '#3b82f6',
                                borderWidth: 1,
                                cornerRadius: 8,
                                displayColors: false,
                                callbacks: {
                                    title: function(context) {
                                        return 'Portfolio Balance';
                                    },
                                    label: function(context) {
                                        return new Intl.NumberFormat('en-US', {
                                            style: 'currency',
                                            currency: 'USD',
                                            minimumFractionDigits: 2
                                        }).format(context.raw);
                                    }
                                }
                            }
                        },
                        scales: {
                            x: { 
                                display: false,
                                grid: { display: false }
                            },
                            y: { 
                                display: false,
                                grid: { display: false },
                                beginAtZero: false
                            }
                        },
                        elements: {
                            point: { radius: 0 }
                        },
                        animation: {
                            duration: 750,
                            easing: 'easeInOutQuart'
                        }
                    }
                });
            }

            // Initialize live trading chart
            const liveCanvas = document.getElementById(`${asset}-live-chart`);
            if (liveCanvas) {
                this.charts[`${asset}-live`] = new Chart(liveCanvas, {
                    type: 'line',
                    data: {
                        labels: [],
                        datasets: [{
                            label: 'Live Portfolio',
                            data: [],
                            borderColor: '#f59e0b',
                            backgroundColor: 'rgba(245, 158, 11, 0.15)',
                            tension: 0.3,
                            fill: true,
                            borderWidth: 2,
                            pointRadius: 0,
                            pointHoverRadius: 4,
                            pointHoverBorderWidth: 2,
                            pointHoverBackgroundColor: '#f59e0b',
                            pointHoverBorderColor: '#ffffff'
                        }]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        interaction: {
                            intersect: false,
                            mode: 'index'
                        },
                        plugins: {
                            legend: { display: false },
                            tooltip: {
                                backgroundColor: 'rgba(26, 31, 46, 0.95)',
                                titleColor: '#ffffff',
                                bodyColor: '#94a3b8',
                                borderColor: '#f59e0b',
                                borderWidth: 1,
                                cornerRadius: 8,
                                displayColors: false,
                                callbacks: {
                                    title: function(context) {
                                        return 'Live Portfolio Balance';
                                    },
                                    label: function(context) {
                                        return new Intl.NumberFormat('en-US', {
                                            style: 'currency',
                                            currency: 'USD',
                                            minimumFractionDigits: 2
                                        }).format(context.raw);
                                    }
                                }
                            }
                        },
                        scales: {
                            x: { 
                                display: false,
                                grid: { display: false }
                            },
                            y: { 
                                display: false,
                                grid: { display: false },
                                beginAtZero: false
                            }
                        },
                        elements: {
                            point: { radius: 0 }
                        },
                        animation: {
                            duration: 750,
                            easing: 'easeInOutQuart'
                        }
                    }
                });
            }
        });
    }

    async loadInitialData() {
        try {
            // Load data for all assets
            await Promise.all([
                this.updateGlobalStats(),
                this.updateAllAssets(),
                this.checkBrokerConnections()
            ]);
            
            this.logActivity('System', 'Initial data loaded successfully');
        } catch (error) {
            console.error('Failed to load initial data:', error);
            this.logActivity('System', 'Failed to load initial data', 'error');
        }
    }

    startPeriodicUpdates() {
        this.updateTimer = setInterval(async () => {
            try {
                await Promise.all([
                    this.updateGlobalStats(),
                    this.updateAllAssets()
                ]);
                
                // Reset reconnect attempts on successful update
                if (!this.isConnected) {
                    this.isConnected = true;
                    this.reconnectAttempts = 0;
                    this.logActivity('System', 'Connection restored', 'success');
                    this.showSystemMessage('Connection restored', 'success');
                }
                this.lastSuccessfulUpdate = Date.now();
                
            } catch (error) {
                console.error('Periodic update failed:', error);
                this.handleConnectionError(error);
            }
        }, this.updateInterval);
    }

    stopPeriodicUpdates() {
        if (this.updateTimer) {
            clearInterval(this.updateTimer);
            this.updateTimer = null;
        }
    }

    async updateGlobalStats() {
        try {
            const response = await fetch(`${this.baseUrl}/stats/global`);
            const data = await response.json();

            // Update global statistics
            this.updateElement('totalPnl', this.formatCurrency(data.total_pnl || 0));
            this.updateElement('activeAssets', `${data.active_assets || 0}/4`);
            this.updateElement('totalPositions', data.total_positions || 0);

            // Update global status
            const globalStatus = document.querySelector('.system-status');
            if (globalStatus) {
                if (data.system_online) {
                    globalStatus.className = 'system-status online';
                    const statusText = globalStatus.querySelector('span');
                    if (statusText) statusText.textContent = 'System Online';
                } else {
                    globalStatus.className = 'system-status';
                    const statusText = globalStatus.querySelector('span');
                    if (statusText) statusText.textContent = 'System Offline';
                }
            }

        } catch (error) {
            console.error('Failed to update global stats:', error);
            // Show offline status on error
            const globalStatus = document.querySelector('.system-status');
            if (globalStatus) {
                globalStatus.className = 'system-status';
                const statusText = globalStatus.querySelector('span');
                if (statusText) statusText.textContent = 'System Offline';
            }
        }
    }

    async updateAllAssets() {
        const updatePromises = this.assets.map(asset => this.updateAsset(asset));
        await Promise.all(updatePromises);
    }

    async updateAsset(asset) {
        try {
            const [statusResponse, positionsResponse, strategyResponse] = await Promise.all([
                fetch(`${this.baseUrl}/asset/${asset}/status`),
                fetch(`${this.baseUrl}/asset/${asset}/positions`),
                fetch(`${this.baseUrl}/asset/${asset}/strategies`)
            ]);

            const status = await statusResponse.json();
            const positions = await positionsResponse.json();
            const strategies = await strategyResponse.json();

            // Update connection status
            this.updateConnectionStatus(asset, status.connection_status);

            // Update wallet balances
            this.updateWallet(asset, 'paper', status.paper_wallet);
            this.updateWallet(asset, 'live', status.live_wallet);

            // Update charts
            this.updateChart(asset, 'paper', status.paper_wallet);
            this.updateChart(asset, 'live', status.live_wallet);

            // Update positions
            this.updatePositions(asset, positions);

            // Update strategy status
            this.updateStrategyStatus(asset, strategies);

            // Update control buttons based on status
            this.updateAssetControls(asset, status);

        } catch (error) {
            console.error(`Failed to update asset ${asset}:`, error);
            this.updateConnectionStatus(asset, 'offline');
            this.showNoData(asset);
        }
    }

    updateConnectionStatus(asset, status) {
        const statusElement = document.getElementById(`${asset}-connection-status`);
        if (!statusElement) return;

        let statusClass = 'offline';
        let statusText = 'Offline';
        
        switch(status) {
            case 'connected':
                statusClass = 'online';
                statusText = 'Connected';
                break;
            case 'auth_failed':
                statusClass = 'auth-error';
                statusText = 'Auth Error';
                break;
            case 'offline':
            default:
                statusClass = 'offline';
                statusText = 'Offline';
                break;
        }

        statusElement.className = `connection-status ${statusClass}`;
        
        const statusTextElement = statusElement.querySelector('span');
        if (statusTextElement) {
            statusTextElement.textContent = statusText;
        }
    }

    updateWallet(asset, type, walletData) {
        if (!walletData) {
            this.updateElement(`${asset}-${type}-balance`, 'Total: $0.00 | Available: $0.00 | Used: $0.00');
            this.updateElement(`${asset}-${type}-change`, 'No data');
            return;
        }

        const balance = walletData.balance || 0;
        const pnl = walletData.pnl || 0;
        const pnlPercent = walletData.pnl_percent || 0;
        
        // Calculate used amount from open positions (if available)
        const usedInPositions = walletData.used_in_positions || 0;
        const availableBalance = walletData.available_balance || (balance - usedInPositions);

        // Create enhanced balance display
        const balanceDisplay = `Total: ${this.formatCurrency(balance)} | Available: ${this.formatCurrency(availableBalance)} | Used: ${this.formatCurrency(usedInPositions)}`;
        
        this.updateElement(`${asset}-${type}-balance`, balanceDisplay);
        
        const changeElement = document.getElementById(`${asset}-${type}-change`);
        if (changeElement) {
            const changeText = `${pnl >= 0 ? '+' : ''}${this.formatCurrency(pnl)} (${pnlPercent >= 0 ? '+' : ''}${pnlPercent.toFixed(2)}%)`;
            changeElement.textContent = changeText;
            changeElement.className = `balance-change ${pnl >= 0 ? 'positive' : 'negative'}`;
        }
    }

    updateChart(asset, type, walletData) {
        const chartKey = `${asset}-${type}`;
        const chart = this.charts[chartKey];
        
        if (!chart || !walletData || !walletData.history) return;

        const history = walletData.history.slice(-20); // Last 20 data points
        const data = history.map(point => point.balance);
        const labels = history.map((point, i) => {
            // Create time-based labels (simplified)
            const now = new Date();
            const minutesAgo = (history.length - i) * 5; // Assume 5-minute intervals
            const time = new Date(now.getTime() - minutesAgo * 60000);
            return time.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' });
        });

        // Determine chart color based on performance
        const startingBalance = data[0] || 1000;
        const currentBalance = data[data.length - 1] || startingBalance;
        const isProfit = currentBalance >= startingBalance;
        
        // Update colors dynamically
        const colors = this.getChartColors(type, isProfit);
        chart.data.datasets[0].borderColor = colors.border;
        chart.data.datasets[0].backgroundColor = colors.background;
        chart.data.datasets[0].pointHoverBackgroundColor = colors.border;

        chart.data.labels = labels;
        chart.data.datasets[0].data = data;
        
        // Smooth update animation
        chart.update('active');
    }
    
    getChartColors(type, isProfit) {
        if (type === 'paper') {
            return isProfit ? 
                { border: '#10b981', background: 'rgba(16, 185, 129, 0.15)' } :
                { border: '#ef4444', background: 'rgba(239, 68, 68, 0.15)' };
        } else {
            return isProfit ? 
                { border: '#10b981', background: 'rgba(16, 185, 129, 0.15)' } :
                { border: '#ef4444', background: 'rgba(239, 68, 68, 0.15)' };
        }
    }

    updatePositions(asset, positionsData) {
        // Handle the nested position data structure
        const paperData = positionsData.paper || {};
        const liveData = positionsData.live || {};
        
        const paperPositions = paperData.positions || [];
        const livePositions = liveData.positions || [];
        
        // Update paper trading metrics
        const paperPositionElement = document.getElementById(`${asset}-paper-positions`);
        const paperDailyPnlElement = document.getElementById(`${asset}-paper-dailypnl`);
        const paperPositionsListElement = document.getElementById(`${asset}-paper-positions-list`);
        
        if (paperPositionElement) paperPositionElement.textContent = paperPositions.length;
        if (paperDailyPnlElement) {
            const dailyPnl = paperData.daily_pnl || 0;
            paperDailyPnlElement.textContent = this.formatCurrency(dailyPnl);
            paperDailyPnlElement.className = `metric-value ${dailyPnl >= 0 ? 'positive' : 'negative'}`;
        }
        if (paperPositionsListElement) {
            this.updatePositionsList(paperPositionsListElement, paperPositions);
        }
        
        // Update live trading metrics
        const livePositionElement = document.getElementById(`${asset}-live-positions`);
        const liveDailyPnlElement = document.getElementById(`${asset}-live-dailypnl`);
        const livePositionsListElement = document.getElementById(`${asset}-live-positions-list`);
        
        if (livePositionElement) livePositionElement.textContent = livePositions.length;
        if (liveDailyPnlElement) {
            const dailyPnl = liveData.daily_pnl || 0;
            liveDailyPnlElement.textContent = this.formatCurrency(dailyPnl);
            liveDailyPnlElement.className = `metric-value ${dailyPnl >= 0 ? 'positive' : 'negative'}`;
        }
        if (livePositionsListElement) {
            this.updatePositionsList(livePositionsListElement, livePositions);
        }
    }
    
    updatePositionsList(container, positions) {
        // Check if container is a table
        const isTable = container.tagName === 'TABLE';
        const tbody = isTable ? container.querySelector('tbody') : null;
        
        if (!positions || positions.length === 0) {
            if (isTable && tbody) {
                tbody.innerHTML = '<tr class="no-positions"><td colspan="7">No active positions</td></tr>';
            } else {
                container.innerHTML = '<div class="no-positions">No active positions</div>';
            }
            return;
        }
        
        if (isTable && tbody) {
            // Render as table rows
            tbody.innerHTML = '';
            
            positions.forEach(position => {
                const pnl = position.pnl || 0;
                const pnlPct = position.pnl_pct || 0;
                const pnlClass = pnl >= 0 ? 'positive' : 'negative';
                const sideClass = position.side ? position.side.toLowerCase() : '';
                
                // Calculate position value in USD
                const size = position.size || 0;
                const entryPrice = position.entry_price || 0;
                const positionValue = size * entryPrice;
                
                // Calculate duration using timestamp
                const duration = position.duration || this.calculateDurationFromNow(position.timestamp);
                
                // TP/SL values
                const takeProfit = position.take_profit || 0;
                const stopLoss = position.stop_loss || 0;
                
                const row = document.createElement('tr');
                row.innerHTML = `
                    <td class="position-symbol">${position.symbol || 'N/A'}</td>
                    <td class="position-side ${sideClass}">${(position.side || 'N/A').toUpperCase()}</td>
                    <td class="position-size">${this.formatCurrency(positionValue)}</td>
                    <td class="entry-price">${this.formatPrice(position.entry_price || 0)}</td>
                    <td class="current-price">${this.formatPrice(position.current_price || 0)}</td>
                    <td class="position-pnl ${pnlClass}">${pnl >= 0 ? '+' : ''}${this.formatCurrency(pnl)} (${pnlPct >= 0 ? '+' : ''}${pnlPct.toFixed(2)}%)</td>
                    <td class="take-profit">${this.formatPrice(takeProfit)}</td>
                    <td class="stop-loss">${this.formatPrice(stopLoss)}</td>
                    <td class="position-duration">${duration}</td>
                `;
                
                // Store position data for sorting
                row.positionData = {
                    symbol: position.symbol || '',
                    side: position.side || '',
                    size: positionValue,
                    entry_price: parseFloat(position.entry_price) || 0,
                    current_price: parseFloat(position.current_price) || 0,
                    pnl: pnl,
                    take_profit: parseFloat(takeProfit) || 0,
                    stop_loss: parseFloat(stopLoss) || 0,
                    duration: duration
                };
                
                tbody.appendChild(row);
            });
            
            // Setup sorting for positions table
            this.setupPositionsTableSorting(container);
            
        } else {
            // Fallback to original card-based layout
            const positionsHtml = positions.map(position => {
                const pnl = position.pnl || 0;
                const pnlPct = position.pnl_pct || 0;
                const pnlClass = pnl >= 0 ? 'positive' : 'negative';
                
                return `
                    <div class="position-item">
                        <div class="position-header">
                            <span class="position-symbol">${position.symbol}</span>
                            <span class="position-side ${position.side}">${position.side.toUpperCase()}</span>
                        </div>
                        <div class="position-details">
                            <div class="position-detail">
                                <span class="position-label">Entry Price</span>
                                <span class="position-value">${this.formatCurrency(position.entry_price || 0)}</span>
                            </div>
                            <div class="position-detail">
                                <span class="position-label">Current Price</span>
                                <span class="position-value">${this.formatCurrency(position.current_price || 0)}</span>
                            </div>
                            <div class="position-detail">
                                <span class="position-label">P&L USD</span>
                                <span class="position-value position-pnl ${pnlClass}">
                                    ${pnl >= 0 ? '+' : ''}${this.formatCurrency(pnl)}
                                </span>
                            </div>
                            <div class="position-detail">
                                <span class="position-label">P&L %</span>
                                <span class="position-value position-pnl ${pnlClass}">
                                    ${pnlPct >= 0 ? '+' : ''}${pnlPct.toFixed(2)}%
                                </span>
                            </div>
                        </div>
                    </div>
                `;
            }).join('');
            
            container.innerHTML = positionsHtml;
        }
    }

    updateStrategyStatus(asset, strategyData) {
        const statusElement = document.getElementById(`${asset}-strategy-status`);
        if (!statusElement) return;

        // Use new strategy development data structure
        const summary = strategyData.summary || {};
        
        const developing = summary.developing || 0;
        const pendingValidation = summary.pending_validation || 0;
        const validated = summary.validated || 0;
        const live = summary.live || 0;
        const total = summary.total || 0;

        let statusText = '';
        if (total === 0) {
            statusText = 'No strategies yet';
        } else {
            const parts = [];
            if (developing > 0) {
                parts.push(`${developing} developing`);
            }
            if (pendingValidation > 0) {
                parts.push(`${pendingValidation} pending validation`);
            }
            if (validated > 0) {
                parts.push(`${validated} validated`);
            }
            if (live > 0) {
                parts.push(`${live} live approved`);
            }
            statusText = parts.join(', ') || `${total} strategies`;
        }

        statusElement.textContent = statusText;

        // Update paper trading specific metrics (strategy development section)
        const paperStrategiesElement = document.getElementById(`${asset}-paper-strategies`);
        if (paperStrategiesElement) {
            paperStrategiesElement.textContent = developing + pendingValidation;
        }

        // Update live trading specific metrics (approved strategies section)
        const liveStrategiesElement = document.getElementById(`${asset}-live-strategies`);
        if (liveStrategiesElement) {
            liveStrategiesElement.textContent = live;
        }
    }

    updateAssetControls(asset, status) {
        const paperBtn = document.getElementById(`${asset}-paper-start`);
        const liveBtn = document.getElementById(`${asset}-live-start`);
        const killBtn = document.getElementById(`${asset}-kill`);
        const paperStatus = document.getElementById(`${asset}-paper-status`);
        const liveStatus = document.getElementById(`${asset}-live-status`);

        if (!paperBtn || !liveBtn || !killBtn) return;

        // Update paper trading button and status
        if (status.paper_trading_active) {
            paperBtn.textContent = 'STOP';
            paperBtn.className = 'btn btn-start active';
            this.updateTradingStatus(paperStatus, 'learning', 'Learning');
        } else {
            paperBtn.textContent = 'START';
            paperBtn.className = 'btn btn-start';
            this.updateTradingStatus(paperStatus, 'idle', 'Idle');
        }

        // Update live trading button and status
        if (status.live_trading_active) {
            liveBtn.textContent = 'STOP';
            liveBtn.className = 'btn btn-start active';
            liveBtn.disabled = false;
            this.updateTradingStatus(liveStatus, 'active', 'Live Trading');
        } else {
            liveBtn.textContent = 'START';
            liveBtn.className = 'btn btn-start';
            liveBtn.disabled = !status.live_trading_approved;
            this.updateTradingStatus(liveStatus, 'idle', 'Idle');
        }

        // Update kill switch
        if (status.kill_switch_active) {
            killBtn.textContent = 'KILLED';
            killBtn.className = 'btn btn-kill disabled';
            killBtn.disabled = true;
            this.updateTradingStatus(paperStatus, 'error', 'Killed');
            this.updateTradingStatus(liveStatus, 'error', 'Killed');
        } else {
            killBtn.textContent = 'KILL';
            killBtn.className = 'btn btn-kill';
            killBtn.disabled = false;
        }
    }
    
    updateTradingStatus(statusElement, statusClass, statusText) {
        if (!statusElement) return;
        
        const indicator = statusElement.querySelector('.status-indicator');
        const text = statusElement.querySelector('.status-text');
        
        if (indicator) {
            indicator.className = `status-indicator ${statusClass}`;
        }
        if (text) {
            text.textContent = statusText;
        }
    }

    showNoData(asset) {
        // Show no data state for the asset
        this.updateElement(`${asset}-paper-balance-value`, 'No data');
        this.updateElement(`${asset}-live-balance-value`, 'No data');
        this.updateElement(`${asset}-paper-change`, 'No data');
        this.updateElement(`${asset}-live-change`, 'No data');
        this.updateElement(`${asset}-position-value`, '0');
        this.updateElement(`${asset}-strategies-value`, '0');
        this.updateElement(`${asset}-winrate-value`, '0%');
        this.updateElement(`${asset}-dailypnl-value`, '$0.00');
    }

    async checkBrokerConnections() {
        try {
            console.log('Checking broker connections...');
            
            // Update server status
            this.updateServerStatus('checking', 'Checking...');
            
            // First test basic server connectivity
            const pingResponse = await fetch(`${this.baseUrl}/ping`);
            if (!pingResponse.ok) {
                throw new Error(`Server ping failed with status ${pingResponse.status}`);
            }
            
            this.updateServerStatus('online', 'Online');
            console.log('Server is responding, checking broker status...');
            
            const response = await fetch(`${this.baseUrl}/brokers/status`);
            
            if (response.ok) {
                const data = await response.json();
                console.log('Broker status data:', data);

                // Update status bar with actual broker statuses
                const bybitStatus = data.bybit_status || 'offline';
                const ibkrStatus = data.ibkr_status || 'offline';
                
                this.updateBybitStatus(bybitStatus);
                this.updateIbkrStatus(ibkrStatus);
                
                // Show paper trading as connected since server is responding
                this.updateConnectionStatus('crypto', 'connected');
                this.updateConnectionStatus('futures', 'connected'); 
                this.updateConnectionStatus('forex', 'connected');
                this.updateConnectionStatus('forex_options', 'connected');
                
                // Handle errors
                if (data.bybit_error) {
                    this.showError(`Bybit: ${data.bybit_error}`);
                    this.logActivity('BYBIT', `Live trading issue: ${data.bybit_error}`, 'warning');
                } else if (bybitStatus === 'connected') {
                    this.logActivity('BYBIT', 'Live trading connection verified', 'success');
                }
                
                this.logActivity('SERVER', 'Connection verified', 'success');
            } else {
                throw new Error(`Broker status API responded with ${response.status}`);
            }

        } catch (error) {
            console.error('Failed to check broker connections:', error);
            this.updateServerStatus('offline', 'Offline');
            this.updateBybitStatus('offline');
            this.updateIbkrStatus('offline');
            
            // Set all as offline on error
            this.assets.forEach(asset => {
                this.updateConnectionStatus(asset, 'offline');
            });
            
            this.showError(`Server connection failed: ${error.message}`);
            this.logActivity('SERVER', `Connection failed: ${error.message}`, 'error');
        }
    }
    
    updateServerStatus(status, text) {
        const statusElement = document.getElementById('server-status');
        if (statusElement) {
            statusElement.className = `status-indicator ${status}`;
            const textElement = statusElement.querySelector('span');
            if (textElement) textElement.textContent = text;
        }
    }
    
    updateBybitStatus(status) {
        const statusElement = document.getElementById('bybit-status');
        if (statusElement) {
            const statusClass = status === 'connected' ? 'online' : 
                               status === 'auth_failed' ? 'auth-error' : 'offline';
            const statusText = status === 'connected' ? 'Connected' :
                              status === 'auth_failed' ? 'Auth Error' : 'Offline';
                              
            statusElement.className = `status-indicator ${statusClass}`;
            const textElement = statusElement.querySelector('span');
            if (textElement) textElement.textContent = statusText;
        }
    }
    
    updateIbkrStatus(status) {
        const statusElement = document.getElementById('ibkr-status');
        if (statusElement) {
            const statusClass = status === 'connected' ? 'online' : 'offline';
            const statusText = status === 'connected' ? 'Connected' : 'Offline';
                              
            statusElement.className = `status-indicator ${statusClass}`;
            const textElement = statusElement.querySelector('span');
            if (textElement) textElement.textContent = statusText;
        }
    }
    
    showError(message) {
        const errorDisplay = document.getElementById('error-display');
        const errorMessage = document.getElementById('error-message');
        
        if (errorDisplay && errorMessage) {
            errorMessage.textContent = message;
            errorDisplay.style.display = 'flex';
            
            // Auto-hide after 10 seconds
            setTimeout(() => {
                this.hideError();
            }, 10000);
        }
    }
    
    hideError() {
        const errorDisplay = document.getElementById('error-display');
        if (errorDisplay) {
            errorDisplay.style.display = 'none';
        }
    }

    // Global functions for HTML onclick handlers
    async startAssetTrading(asset, mode) {
        const button = document.getElementById(`${asset}-${mode}-start`);
        
        try {
            // Immediate visual feedback
            if (button) {
                button.className = 'btn btn-start loading';
                button.textContent = 'Starting...';
            }
            
            // Update status immediately
            const statusElement = document.getElementById(`${asset}-${mode}-status`);
            this.updateTradingStatus(statusElement, 'learning', 'Starting...');
            
            // Log activity immediately
            this.logActivity(asset.toUpperCase(), `Starting ${mode} trading...`, 'info');

            const response = await fetch(`${this.baseUrl}/asset/${asset}/start/${mode}`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' }
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const data = await response.json();
            this.logActivity(asset.toUpperCase(), `${mode} trading started successfully`, 'success');
            
            // Update button to active state
            if (button) {
                button.className = 'btn btn-start active';
                button.textContent = 'STOP';
            }
            
            // Update status to learning
            this.updateTradingStatus(statusElement, 'learning', 'Learning');
            
            // Update the asset data
            await this.updateAsset(asset);

        } catch (error) {
            console.error(`Failed to start ${asset} ${mode} trading:`, error);
            this.logActivity(asset.toUpperCase(), `Failed to start ${mode} trading: ${error.message}`, 'error');
            
            // Reset button on error
            if (button) {
                button.className = 'btn btn-start';
                button.textContent = 'START';
            }
            
            // Reset status on error
            const statusElement = document.getElementById(`${asset}-${mode}-status`);
            this.updateTradingStatus(statusElement, 'error', 'Error');
        }
    }

    async stopAssetTrading(asset, mode) {
        try {
            const response = await fetch(`${this.baseUrl}/asset/${asset}/stop/${mode}`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' }
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const data = await response.json();
            this.logActivity(asset.toUpperCase(), `${mode} trading stopped`);
            
            // Update the asset immediately
            await this.updateAsset(asset);

        } catch (error) {
            console.error(`Failed to stop ${asset} ${mode} trading:`, error);
            this.logActivity(asset.toUpperCase(), `Failed to stop ${mode} trading`, 'error');
        }
    }

    async killAssetTrading(asset) {
        if (!confirm(`Are you sure you want to activate the kill switch for ${asset.toUpperCase()}? This will immediately stop all trading and close positions.`)) {
            return;
        }

        try {
            const response = await fetch(`${this.baseUrl}/asset/${asset}/kill`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' }
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const data = await response.json();
            this.logActivity(asset.toUpperCase(), 'Kill switch activated', 'warning');
            
            // Update the asset immediately
            await this.updateAsset(asset);

        } catch (error) {
            console.error(`Failed to activate kill switch for ${asset}:`, error);
            this.logActivity(asset.toUpperCase(), 'Failed to activate kill switch', 'error');
        }
    }

    async emergencyStopAll() {
        if (!confirm('EMERGENCY STOP: This will immediately stop ALL trading across all assets and close all positions. Are you sure?')) {
            return;
        }

        try {
            const response = await fetch(`${this.baseUrl}/emergency/stop`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' }
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            this.logActivity('SYSTEM', 'EMERGENCY STOP ACTIVATED', 'error');
            
            // Update all assets
            await this.updateAllAssets();

        } catch (error) {
            console.error('Failed to activate emergency stop:', error);
            this.logActivity('SYSTEM', 'Emergency stop failed', 'error');
        }
    }

    logActivity(asset, message, type = 'info') {
        const activityList = document.getElementById('activityList');
        if (!activityList) return;

        // Remove "No activity yet" message
        const noData = activityList.querySelector('.no-data');
        if (noData) {
            noData.remove();
        }

        const activityItem = document.createElement('div');
        activityItem.className = 'activity-item';
        
        const now = new Date();
        const timeString = now.toLocaleTimeString();

        activityItem.innerHTML = `
            <div class="activity-time">${timeString}</div>
            <div class="activity-message">
                <span class="activity-asset">[${asset}]</span> ${message}
            </div>
        `;

        // Add type-specific styling
        if (type === 'error') {
            activityItem.style.borderLeft = '3px solid var(--danger-color)';
        } else if (type === 'warning') {
            activityItem.style.borderLeft = '3px solid var(--warning-color)';
        } else if (type === 'success') {
            activityItem.style.borderLeft = '3px solid var(--success-color)';
        }

        activityList.insertBefore(activityItem, activityList.firstChild);

        // Keep only last 50 activities
        while (activityList.children.length > 50) {
            activityList.removeChild(activityList.lastChild);
        }
    }

    updateElement(id, value, className = null) {
        const element = document.getElementById(id);
        if (element) {
            element.textContent = value;
            if (className !== null) {
                element.className = className;
            }
        }
    }

    formatCurrency(value) {
        return new Intl.NumberFormat('en-US', {
            style: 'currency',
            currency: 'USD',
            minimumFractionDigits: 2,
            maximumFractionDigits: 2
        }).format(value);
    }

    destroy() {
        this.stopPeriodicUpdates();
        
        // Destroy charts
        Object.values(this.charts).forEach(chart => {
            if (chart) chart.destroy();
        });
    }

    // Analytics Methods
    async loadAnalyticsData(asset, mode) {
        try {
            console.log(`Loading analytics data for ${asset} ${mode}`);
            
            const response = await fetch(`${this.baseUrl}/analytics/${asset}/${mode}/reward-summary`);
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            const data = await response.json();
            console.log(`Analytics data received:`, data);
            
            // Update overview metrics
            this.updateElement(`${asset}-${mode}-total-trades`, data.trading_metrics?.total_trades || 0);
            this.updateElement(`${asset}-${mode}-win-rate`, `${data.trading_metrics?.win_rate || 0}%`);
            this.updateElement(`${asset}-${mode}-profit-factor`, data.trading_metrics?.profit_factor || '0.00');
            this.updateElement(`${asset}-${mode}-roi`, `${data.overall_performance?.roi_pct || 0}%`);
            this.updateElement(`${asset}-${mode}-avg-win`, this.formatCurrency(data.trading_metrics?.avg_win || 0));
            this.updateElement(`${asset}-${mode}-avg-loss`, this.formatCurrency(data.trading_metrics?.avg_loss || 0));
            this.updateElement(`${asset}-${mode}-max-win`, this.formatCurrency(data.trading_metrics?.max_win || 0));
            this.updateElement(`${asset}-${mode}-max-loss`, this.formatCurrency(data.trading_metrics?.max_loss || 0));
            
            // Update reward system metrics if available
            if (data.reward_system_metrics) {
                const rewards = data.reward_system_metrics;
                
                // Update reward points with color coding
                const totalRewardPoints = rewards.total_reward_points || 0;
                const avgRewardPerTrade = rewards.avg_reward_per_trade || 0;
                
                const rewardPointsElement = document.getElementById(`${asset}-${mode}-reward-points`);
                if (rewardPointsElement) {
                    rewardPointsElement.textContent = totalRewardPoints.toFixed(1);
                    rewardPointsElement.className = `metric-value ${totalRewardPoints >= 0 ? 'reward-positive' : 'reward-negative'}`;
                }
                
                const avgRewardElement = document.getElementById(`${asset}-${mode}-avg-reward`);
                if (avgRewardElement) {
                    avgRewardElement.textContent = avgRewardPerTrade.toFixed(2);
                    avgRewardElement.className = `metric-value ${avgRewardPerTrade >= 0 ? 'reward-positive' : 'reward-negative'}`;
                }
                
                this.updateElement(`${asset}-${mode}-consecutive-sl`, rewards.consecutive_sl_hits || 0);
                
                // Show/hide reward section based on availability
                const rewardSection = document.getElementById(`${asset}-${mode}-reward-section`);
                if (rewardSection) {
                    rewardSection.style.display = 'block';
                }
            }
            
        } catch (error) {
            console.error(`Failed to load analytics data for ${asset} ${mode}:`, error);
            this.logActivity(asset.toUpperCase(), `Failed to load analytics data: ${error.message}`, 'error');
        }
    }

    async loadTradeHistory(asset, mode, forceRefresh = false) {
        try {
            console.log(`Loading trade history for ${asset} ${mode}`);
            
            const response = await fetch(`${this.baseUrl}/analytics/${asset}/${mode}/detailed-trades`);
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            const data = await response.json();
            console.log(`Trade history data received:`, data);
            
            const tradeHistoryTable = document.getElementById(`${asset}-${mode}-trade-history`);
            if (!tradeHistoryTable) return;
            
            const tbody = tradeHistoryTable.querySelector('tbody');
            if (!tbody) return;
            
            // Clear existing content
            tbody.innerHTML = '';
            
            if (!data.trades || data.trades.length === 0) {
                tbody.innerHTML = '<tr class="no-trades"><td colspan="10">No completed trades yet</td></tr>';
                return;
            }
            
            // Store original data for sorting and filtering
            this.tradeData = this.tradeData || {};
            this.tradeData[`${asset}-${mode}`] = data.trades;
            
            // Render table rows
            this.renderTradeHistoryTable(asset, mode, data.trades);
            
            // Setup table sorting and filtering
            this.setupTradeHistoryControls(asset, mode);
            
            if (forceRefresh) {
                this.logActivity(asset.toUpperCase(), `Trade history refreshed (${data.trades.length} trades)`, 'info');
            }
            
        } catch (error) {
            console.error(`Failed to load trade history for ${asset} ${mode}:`, error);
            this.logActivity(asset.toUpperCase(), `Failed to load trade history: ${error.message}`, 'error');
            
            const tradeHistoryList = document.getElementById(`${asset}-${mode}-trade-history`);
            if (tradeHistoryList) {
                tradeHistoryList.innerHTML = '<div class="no-trades">Error loading trade history</div>';
            }
        }
    }

    async addTradingStatisticsHeader(container, asset, mode) {
        try {
            // Get comprehensive analytics data for the header
            const response = await fetch(`${this.baseUrl}/analytics/${asset}/${mode}/reward-summary`);
            if (!response.ok) return;
            
            const data = await response.json();
            const metrics = data.trading_metrics || {};
            const performance = data.overall_performance || {};
            
            const headerDiv = document.createElement('div');
            headerDiv.className = 'trading-stats-header';
            
            // Calculate average position size and total volume
            const symbolBreakdown = data.symbol_breakdown || {};
            let totalVolume = 0;
            let totalTrades = metrics.total_trades || 0;
            
            Object.values(symbolBreakdown).forEach(symbol => {
                totalVolume += symbol.total_volume || 0;
            });
            
            const avgPositionSize = totalTrades > 0 ? (totalVolume / totalTrades) : 0;
            
            headerDiv.innerHTML = `
                <div class="overall-stats-line">
                    [Trades: ${totalTrades} | 
                    Win%: ${(metrics.win_rate || 0).toFixed(1)} | 
                    PnL: ${performance.roi_pct >= 0 ? '+' : ''}${(performance.roi_pct || 0).toFixed(1)}% | 
                    Sharpe: ${(performance.sharpe_ratio || 0).toFixed(2)} | 
                    Avg Size: ${this.formatCurrency(avgPositionSize)} | 
                    Total Vol: ${this.formatCurrency(totalVolume)} | 
                    Avg Win: +${(Math.abs(metrics.avg_win) || 0).toFixed(2)}% | 
                    Max DD: ${(metrics.max_loss || 0).toFixed(1)}%]
                </div>
            `;
            
            container.appendChild(headerDiv);
        } catch (error) {
            console.warn('Could not load trading statistics header:', error);
        }
    }

    formatPrice(price) {
        const numPrice = parseFloat(price) || 0;
        if (numPrice >= 1000) {
            return numPrice.toLocaleString('en-US', { minimumFractionDigits: 0, maximumFractionDigits: 0 });
        } else if (numPrice >= 1) {
            return numPrice.toFixed(2);
        } else {
            return numPrice.toFixed(4);
        }
    }

    estimateStopLoss(entryPrice, side) {
        const price = parseFloat(entryPrice) || 0;
        const slPercent = 0.03; // 3% stop loss
        
        if (side === 'BUY') {
            return price * (1 - slPercent);
        } else {
            return price * (1 + slPercent);
        }
    }

    estimateTakeProfit(entryPrice, side) {
        const price = parseFloat(entryPrice) || 0;
        const tpPercent = 0.05; // 5% take profit
        
        if (side === 'BUY') {
            return price * (1 + tpPercent);
        } else {
            return price * (1 - tpPercent);
        }
    }

    calculateDuration(entryTime, exitTime) {
        try {
            const entry = new Date(entryTime);
            const exit = new Date(exitTime);
            const diffMs = exit - entry;
            
            const hours = Math.floor(diffMs / (1000 * 60 * 60));
            const minutes = Math.floor((diffMs % (1000 * 60 * 60)) / (1000 * 60));
            
            if (hours > 0) {
                return `${hours}h${minutes.toString().padStart(2, '0')}m`;
            } else {
                return `${minutes}m`;
            }
        } catch (error) {
            return 'N/A';
        }
    }

    calculateDurationFromNow(entryTime) {
        try {
            if (!entryTime) {
                return 'N/A';
            }
            
            const entry = new Date(entryTime);
            const now = new Date();
            const diffMs = now - entry;
            
            if (diffMs < 0) {
                return 'N/A'; // Entry time in future
            }
            
            const hours = Math.floor(diffMs / (1000 * 60 * 60));
            const minutes = Math.floor((diffMs % (1000 * 60 * 60)) / (1000 * 60));
            const seconds = Math.floor((diffMs % (1000 * 60)) / 1000);
            
            if (hours > 0) {
                return `${hours}h${minutes.toString().padStart(2, '0')}m`;
            } else if (minutes > 0) {
                return `${minutes}m${seconds.toString().padStart(2, '0')}s`;
            } else {
                return `${seconds}s`;
            }
        } catch (error) {
            return 'N/A';
        }
    }

    formatTradeTimestamp(timestamp) {
        try {
            const date = new Date(timestamp);
            const now = new Date();
            const today = new Date(now.getFullYear(), now.getMonth(), now.getDate());
            const tradeDate = new Date(date.getFullYear(), date.getMonth(), date.getDate());
            
            const timeStr = date.toLocaleTimeString('en-US', { 
                hour: '2-digit', 
                minute: '2-digit', 
                second: '2-digit',
                hour12: false 
            });
            
            if (tradeDate.getTime() === today.getTime()) {
                return `${timeStr}`;
            } else {
                const dateStr = date.toLocaleDateString('en-US', { 
                    month: '2-digit', 
                    day: '2-digit' 
                });
                return `${dateStr} ${timeStr}`;
            }
        } catch (error) {
            return 'N/A';
        }
    }

    async loadSymbolPerformance(asset, mode) {
        try {
            console.log(`Loading symbol performance for ${asset} ${mode}`);
            
            const response = await fetch(`${this.baseUrl}/analytics/${asset}/${mode}/symbol-performance`);
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            const data = await response.json();
            console.log(`Symbol performance data received:`, data);
            
            const symbolPerformanceList = document.getElementById(`${asset}-${mode}-symbol-performance`);
            if (!symbolPerformanceList) return;
            
            // Clear existing content
            symbolPerformanceList.innerHTML = '';
            
            if (!data.symbols || Object.keys(data.symbols).length === 0) {
                symbolPerformanceList.innerHTML = '<div class="no-data">No symbol data available</div>';
                return;
            }
            
            // Create symbol items
            Object.entries(data.symbols).forEach(([symbol, performance]) => {
                const symbolItem = document.createElement('div');
                symbolItem.className = 'symbol-item';
                
                const pnlClass = performance.total_pnl >= 0 ? 'positive' : 'negative';
                
                symbolItem.innerHTML = `
                    <div class="symbol-name">${symbol}</div>
                    <div class="symbol-trades">${performance.total_trades} trades</div>
                    <div class="symbol-winrate">${performance.win_rate}%</div>
                    <div class="symbol-pnl ${pnlClass}">${this.formatCurrency(performance.total_pnl)}</div>
                `;
                
                symbolPerformanceList.appendChild(symbolItem);
            });
            
        } catch (error) {
            console.error(`Failed to load symbol performance for ${asset} ${mode}:`, error);
            this.logActivity(asset.toUpperCase(), `Failed to load symbol performance: ${error.message}`, 'error');
            
            const symbolPerformanceList = document.getElementById(`${asset}-${mode}-symbol-performance`);
            if (symbolPerformanceList) {
                symbolPerformanceList.innerHTML = '<div class="no-data">Error loading symbol performance</div>';
            }
        }
    }

    renderTradeHistoryTable(asset, mode, trades) {
        const tradeHistoryTable = document.getElementById(`${asset}-${mode}-trade-history`);
        const tbody = tradeHistoryTable.querySelector('tbody');
        
        // Clear existing rows
        tbody.innerHTML = '';
        
        if (!trades || trades.length === 0) {
            tbody.innerHTML = '<tr class="no-trades"><td colspan="11">No completed trades yet</td></tr>';
            return;
        }
        
        trades.forEach((trade, index) => {
            const pnlUsd = trade.pnl_usd || trade.pnl || 0;
            const pnlPct = trade.pnl_pct || trade.pnl_percentage || 0;
            const pnlClass = pnlUsd >= 0 ? 'positive' : 'negative';
            const sideClass = trade.side === 'BUY' ? 'long' : 'short';
            
            // Format prices with proper precision
            const entryPrice = this.formatPrice(trade.entry_price);
            const exitPrice = this.formatPrice(trade.exit_price);
            
            // Calculate position size and trade amount
            const positionSize = trade.size || 0;
            const tradeAmount = trade.trade_amount_usd || (positionSize * parseFloat(trade.entry_price)) || 0;
            
            // Format timestamps
            const exitTime = this.formatTradeTimestamp(trade.exit_time);
            const entryTime = this.formatTradeTimestamp(trade.entry_time);
            
            // Calculate running balance (if available from trade data)
            const runningBalance = trade.balance || trade.portfolio_balance || trade.balance_after || 0;
            
            // Calculate rewards (if available from trade data or reward system)
            const rewards = trade.reward_points || trade.reward || trade.rewards || this.calculateTradeRewards(trade);
            const rewardClass = rewards >= 0 ? 'positive' : 'negative';
            
            const row = document.createElement('tr');
            row.innerHTML = `
                <td class="trade-symbol">${trade.symbol}</td>
                <td class="trade-side ${sideClass}">${trade.side === 'BUY' ? 'Long' : 'Short'}</td>
                <td class="trade-size">${this.formatCurrency(tradeAmount)}</td>
                <td class="entry-price">${entryPrice}</td>
                <td class="entry-time">${entryTime}</td>
                <td class="exit-price">${exitPrice}</td>
                <td class="trade-timestamp">${exitTime}</td>
                <td class="trade-duration">${trade.duration || this.calculateDuration(trade.entry_time, trade.exit_time)}</td>
                <td class="pnl-pct ${pnlClass}">${pnlPct >= 0 ? '+' : ''}${pnlPct.toFixed(2)}%</td>
                <td class="pnl-usd ${pnlClass}">${pnlUsd >= 0 ? '+' : ''}${this.formatCurrency(pnlUsd)}</td>
                <td class="trade-rewards ${rewardClass}">${rewards >= 0 ? '+' : ''}${rewards.toFixed(1)}</td>
                <td class="running-balance">${runningBalance > 0 ? this.formatCurrency(runningBalance) : 'N/A'}</td>
            `;
            
            // Store trade data on row for sorting
            row.tradeData = trade;
            row.tradeData.timestamp = Date.parse(trade.exit_time) || 0;
            row.tradeData.symbol = trade.symbol;
            row.tradeData.side = trade.side;
            row.tradeData.size = tradeAmount;
            row.tradeData.entry = parseFloat(trade.entry_price) || 0;
            row.tradeData.entry_time = Date.parse(trade.entry_time) || 0;
            row.tradeData.exit = parseFloat(trade.exit_price) || 0;
            row.tradeData.pnl_pct = pnlPct;
            row.tradeData.pnl_usd = pnlUsd;
            row.tradeData.rewards = rewards;
            row.tradeData.balance = runningBalance;
            row.tradeData.duration = trade.duration || this.calculateDuration(trade.entry_time, trade.exit_time);
            
            tbody.appendChild(row);
        });
    }
    
    setupTradeHistoryControls(asset, mode) {
        const tableId = `${asset}-${mode}-trade-history`;
        const searchId = `${asset}-${mode}-search`;
        const filterId = `${asset}-${mode}-trade-filter`;
        
        // Setup column sorting
        this.setupTableSorting(tableId);
        
        // Setup search filter
        const searchInput = document.getElementById(searchId);
        if (searchInput) {
            searchInput.addEventListener('input', (e) => {
                this.filterTradeHistory(asset, mode, e.target.value.toLowerCase());
            });
        }
        
        // Setup win/loss filter
        const filterSelect = document.getElementById(filterId);
        if (filterSelect) {
            filterSelect.addEventListener('change', (e) => {
                this.filterTradeHistoryByType(asset, mode, e.target.value);
            });
        }
    }

    setupTableSorting(tableId) {
        const table = document.getElementById(tableId);
        if (!table) return;
        
        const headers = table.querySelectorAll('th.sortable');
        headers.forEach(header => {
            header.addEventListener('click', () => {
                const column = header.dataset.column;
                const currentDirection = header.classList.contains('sorted-asc') ? 'asc' : 
                                      header.classList.contains('sorted-desc') ? 'desc' : 'none';
                
                // Clear all other sort indicators
                headers.forEach(h => h.classList.remove('sorted-asc', 'sorted-desc'));
                
                // Determine new direction
                let newDirection;
                if (currentDirection === 'none' || currentDirection === 'desc') {
                    newDirection = 'asc';
                    header.classList.add('sorted-asc');
                } else {
                    newDirection = 'desc';
                    header.classList.add('sorted-desc');
                }
                
                // Sort table
                this.sortTable(table, column, newDirection);
            });
        });
    }
    
    sortTable(table, column, direction) {
        const tbody = table.querySelector('tbody');
        const rows = Array.from(tbody.querySelectorAll('tr'));
        
        // Filter out no-data rows
        const dataRows = rows.filter(row => row.tradeData);
        
        if (dataRows.length === 0) return;
        
        dataRows.sort((a, b) => {
            let valueA = a.tradeData[column];
            let valueB = b.tradeData[column];
            
            // Handle different data types
            if (typeof valueA === 'string' && typeof valueB === 'string') {
                valueA = valueA.toLowerCase();
                valueB = valueB.toLowerCase();
            }
            
            if (valueA < valueB) return direction === 'asc' ? -1 : 1;
            if (valueA > valueB) return direction === 'asc' ? 1 : -1;
            return 0;
        });
        
        // Re-append sorted rows
        tbody.innerHTML = '';
        dataRows.forEach(row => tbody.appendChild(row));
        
        // Add no-data row if needed
        if (dataRows.length === 0) {
            tbody.innerHTML = '<tr class="no-trades"><td colspan="12">No completed trades yet</td></tr>';
        }
    }
    
    setupPositionsTableSorting(table) {
        if (!table) return;
        
        const headers = table.querySelectorAll('th.sortable');
        headers.forEach(header => {
            header.addEventListener('click', () => {
                const column = header.dataset.column;
                const currentDirection = header.classList.contains('sorted-asc') ? 'asc' : 
                                      header.classList.contains('sorted-desc') ? 'desc' : 'none';
                
                // Clear all other sort indicators
                headers.forEach(h => h.classList.remove('sorted-asc', 'sorted-desc'));
                
                // Determine new direction
                let newDirection;
                if (currentDirection === 'none' || currentDirection === 'desc') {
                    newDirection = 'asc';
                    header.classList.add('sorted-asc');
                } else {
                    newDirection = 'desc';
                    header.classList.add('sorted-desc');
                }
                
                // Sort positions table
                this.sortPositionsTable(table, column, newDirection);
            });
        });
    }
    
    sortPositionsTable(table, column, direction) {
        const tbody = table.querySelector('tbody');
        const rows = Array.from(tbody.querySelectorAll('tr'));
        
        // Filter out no-data rows
        const dataRows = rows.filter(row => row.positionData);
        
        if (dataRows.length === 0) return;
        
        dataRows.sort((a, b) => {
            let valueA = a.positionData[column];
            let valueB = b.positionData[column];
            
            // Handle different data types
            if (typeof valueA === 'string' && typeof valueB === 'string') {
                valueA = valueA.toLowerCase();
                valueB = valueB.toLowerCase();
            }
            
            if (valueA < valueB) return direction === 'asc' ? -1 : 1;
            if (valueA > valueB) return direction === 'asc' ? 1 : -1;
            return 0;
        });
        
        // Re-append sorted rows
        tbody.innerHTML = '';
        dataRows.forEach(row => tbody.appendChild(row));
        
        // Add no-data row if needed
        if (dataRows.length === 0) {
            tbody.innerHTML = '<tr class="no-positions"><td colspan="9">No active positions</td></tr>';
        }
    }
    
    calculateTradeRewards(trade) {
        // Simple reward calculation based on P&L and other factors
        // This integrates with the existing reward system if available
        const pnlPct = trade.pnl_pct || trade.pnl_percentage || 0;
        const pnlUsd = trade.pnl_usd || trade.pnl || 0;
        
        // Base reward from P&L percentage 
        let reward = pnlPct * 10; // 1% profit = 10 points
        
        // Bonus for larger absolute profits
        if (Math.abs(pnlUsd) > 100) {
            reward += Math.sign(pnlUsd) * 5;
        }
        
        // Penalty for losses
        if (pnlUsd < 0) {
            reward -= Math.abs(pnlPct) * 2; // Extra penalty for losses
        }
        
        return reward;
    }
    
    handleConnectionError(error) {
        this.isConnected = false;
        this.reconnectAttempts++;
        
        console.error(`Connection error (attempt ${this.reconnectAttempts}/${this.maxReconnectAttempts}):`, error);
        
        if (this.reconnectAttempts <= this.maxReconnectAttempts) {
            const remainingAttempts = this.maxReconnectAttempts - this.reconnectAttempts + 1;
            this.logActivity('System', `Server not reachable, attempt to reconnect #${this.reconnectAttempts}`, 'warning');
            this.showSystemMessage(`Server not reachable, attempting to reconnect... (${remainingAttempts} attempts remaining)`, 'warning');
            
            // If too many attempts, switch to longer retry interval
            if (this.reconnectAttempts > 5) {
                this.updateInterval = Math.min(this.updateInterval * 1.5, 15000); // Max 15 seconds
            }
        } else {
            // Max attempts reached
            this.logActivity('System', `Max reconnection attempts reached. Please check your connection.`, 'error');
            this.showSystemMessage(`Connection failed after ${this.maxReconnectAttempts} attempts. Please check your connection.`, 'error');
            this.stopPeriodicUpdates();
        }
    }
    
    showSystemMessage(message, type = 'info') {
        // Create or update system message display
        let messageDiv = document.getElementById('system-message');
        if (!messageDiv) {
            messageDiv = document.createElement('div');
            messageDiv.id = 'system-message';
            messageDiv.className = 'system-message';
            document.body.appendChild(messageDiv);
        }
        
        messageDiv.className = `system-message ${type}`;
        messageDiv.innerHTML = `
            <div class="message-content">
                <i class="fas ${this.getMessageIcon(type)}"></i>
                <span>${message}</span>
                <button class="message-close" onclick="this.parentElement.parentElement.style.display='none'">
                    <i class="fas fa-times"></i>
                </button>
            </div>
        `;
        
        messageDiv.style.display = 'block';
        
        // Auto-hide success messages after 5 seconds
        if (type === 'success') {
            setTimeout(() => {
                messageDiv.style.display = 'none';
            }, 5000);
        }
    }
    
    getMessageIcon(type) {
        switch (type) {
            case 'success': return 'fa-check-circle';
            case 'warning': return 'fa-exclamation-triangle';
            case 'error': return 'fa-exclamation-circle';
            default: return 'fa-info-circle';
        }
    }
    
    filterTradeHistory(asset, mode, searchTerm) {
        const tableId = `${asset}-${mode}-trade-history`;
        const table = document.getElementById(tableId);
        if (!table) return;
        
        const rows = table.querySelectorAll('tbody tr');
        
        rows.forEach(row => {
            if (!row.tradeData) {
                row.style.display = searchTerm === '' ? '' : 'none';
                return;
            }
            
            const symbol = row.tradeData.symbol.toLowerCase();
            const side = row.tradeData.side.toLowerCase();
            
            const matches = symbol.includes(searchTerm) || side.includes(searchTerm);
            row.style.display = matches ? '' : 'none';
        });
    }
    
    filterTradeHistoryByType(asset, mode, filterType) {
        const tableId = `${asset}-${mode}-trade-history`;
        const table = document.getElementById(tableId);
        if (!table) return;
        
        const rows = table.querySelectorAll('tbody tr');
        
        rows.forEach(row => {
            if (!row.tradeData) {
                row.style.display = filterType === 'all' ? '' : 'none';
                return;
            }
            
            const pnl = row.tradeData.pnl_usd;
            let show = true;
            
            switch (filterType) {
                case 'wins':
                    show = pnl >= 0;
                    break;
                case 'losses':
                    show = pnl < 0;
                    break;
                case 'all':
                default:
                    show = true;
                    break;
            }
            
            row.style.display = show ? '' : 'none';
        });
    }
}

// Analytics functions
function toggleAnalytics(asset, mode) {
    console.log(`Toggle analytics called: ${asset} ${mode}`);
    const content = document.getElementById(`${asset}-${mode}-analytics-content`);
    const button = document.querySelector(`#${asset}-${mode}-analytics .btn-analytics`);
    
    if (content && button) {
        const isVisible = content.style.display !== 'none';
        content.style.display = isVisible ? 'none' : 'block';
        button.classList.toggle('expanded', !isVisible);
        
        // Load analytics data when opening
        if (!isVisible && dashboard) {
            dashboard.loadAnalyticsData(asset, mode);
        }
    }
}

function switchTab(prefix, tabType) {
    console.log(`Switch tab called: ${prefix} ${tabType}`);
    
    // Hide all tab contents for this prefix
    const tabContents = document.querySelectorAll(`[id^="${prefix}-"][id$="-tab"]`);
    tabContents.forEach(tab => {
        tab.classList.remove('active');
    });
    
    // Remove active class from all tab buttons for this prefix
    const tabButtons = document.querySelectorAll(`#${prefix}-analytics-content .tab-btn`);
    tabButtons.forEach(btn => {
        btn.classList.remove('active');
    });
    
    // Show selected tab content
    const selectedTab = document.getElementById(`${prefix}-${tabType}-tab`);
    if (selectedTab) {
        selectedTab.classList.add('active');
    }
    
    // Activate selected tab button
    const selectedButton = document.querySelector(`#${prefix}-analytics-content .tab-btn:nth-child(${tabType === 'overview' ? 1 : tabType === 'trades' ? 2 : 3})`);
    if (selectedButton) {
        selectedButton.classList.add('active');
    }
    
    // Load specific data based on tab type
    if (dashboard) {
        const [asset, mode] = prefix.split('-');
        if (tabType === 'trades') {
            dashboard.loadTradeHistory(asset, mode);
        } else if (tabType === 'symbols') {
            dashboard.loadSymbolPerformance(asset, mode);
        } else if (tabType === 'overview') {
            dashboard.loadAnalyticsData(asset, mode);
        }
    }
}

function refreshTradeHistory(asset, mode) {
    console.log(`Refresh trade history called: ${asset} ${mode}`);
    if (dashboard) {
        dashboard.loadTradeHistory(asset, mode, true);
    }
}

// Global functions for HTML onclick handlers
let dashboard;

function startAssetTrading(asset, mode) {
    console.log(`Start asset trading called: ${asset} ${mode}`);
    
    if (dashboard) {
        const button = event?.target || document.getElementById(`${asset}-${mode}-start`);
        const isCurrentlyActive = button && button.textContent.includes('STOP');
        
        if (isCurrentlyActive) {
            dashboard.stopAssetTrading(asset, mode);
        } else {
            dashboard.startAssetTrading(asset, mode);
        }
    } else {
        console.error('Dashboard not initialized');
        alert('Dashboard is still loading. Please wait a moment and try again.');
    }
}

function killAssetTrading(asset) {
    console.log(`Kill asset trading called: ${asset}`);
    
    if (dashboard) {
        dashboard.killAssetTrading(asset);
    } else {
        console.error('Dashboard not initialized');
        alert('Dashboard is still loading. Please wait a moment and try again.');
    }
}

// HTML compatibility functions
function toggleTrading(asset, mode) {
    console.log(`Toggle trading called: ${asset} ${mode}`);
    try {
        startAssetTrading(asset, mode);
    } catch (error) {
        console.error('Toggle trading error:', error);
        alert(`Error starting ${asset} ${mode} trading: ${error.message}`);
    }
}

function killAsset(asset) {
    console.log(`Kill asset called: ${asset}`);
    try {
        killAssetTrading(asset);
    } catch (error) {
        console.error('Kill asset error:', error);
        alert(`Error stopping ${asset} trading: ${error.message}`);
    }
}

function hideError() {
    console.log('Hide error called');
    try {
        if (window.dashboard) {
            window.dashboard.hideError();
        } else {
            const errorDisplay = document.getElementById('error-display');
            if (errorDisplay) {
                errorDisplay.style.display = 'none';
            }
        }
    } catch (error) {
        console.error('Hide error failed:', error);
    }
}

// Initialize dashboard when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    console.log('DOM loaded, initializing dashboard...');
    dashboard = new MultiAssetDashboard();
    window.dashboard = dashboard; // Make it globally accessible
    
    // Force an immediate connection check
    setTimeout(() => {
        if (dashboard) {
            console.log('Running initial connection check...');
            dashboard.checkBrokerConnections();
        }
    }, 1000);
});

// Handle page visibility changes to pause/resume updates
document.addEventListener('visibilitychange', () => {
    if (dashboard) {
        if (document.hidden) {
            dashboard.stopPeriodicUpdates();
        } else {
            dashboard.startPeriodicUpdates();
        }
    }
});

// Cleanup on page unload
window.addEventListener('beforeunload', () => {
    if (dashboard) {
        dashboard.destroy();
    }
});

// Reward Display System
class RewardDisplayManager {
    constructor() {
        this.rewardHistory = [];
        this.maxHistorySize = 50;
        this.sidebarOpen = false;
        this.init();
    }

    init() {
        // Initialize sidebar toggle
        this.setupEventListeners();
        // Load existing reward history from localStorage
        this.loadRewardHistory();
        this.updateHistoryDisplay();
    }

    setupEventListeners() {
        // Click outside modal to close
        document.addEventListener('click', (e) => {
            const modal = document.getElementById('rewardModal');
            if (e.target === modal) {
                this.closeRewardModal();
            }
        });

        // Escape key to close modal
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') {
                this.closeRewardModal();
            }
        });
    }

    showRewardModal(tradeData) {
        const modal = document.getElementById('rewardModal');
        if (!modal) return;

        // Populate modal with trade data
        this.populateRewardModal(tradeData);
        
        // Show modal with animation
        modal.classList.add('show');
        
        // Auto-close after 8 seconds unless user interacts
        this.autoCloseTimer = setTimeout(() => {
            this.closeRewardModal();
        }, 8000);
    }

    populateRewardModal(tradeData) {
        const {
            symbol, side, realizedProfitPct, reward, components,
            entryPrice, exitPrice, qty, leverage
        } = tradeData;

        // Update summary section
        document.getElementById('rewardTotalValue').textContent = 
            reward >= 0 ? `+${reward.toFixed(2)}` : reward.toFixed(2);
        document.getElementById('rewardTotalValue').className = 
            `reward-value ${reward >= 0 ? '' : 'negative'}`;

        document.getElementById('rewardTradeSymbol').textContent = symbol;
        document.getElementById('rewardTradeSide').textContent = side?.toUpperCase() || '-';
        
        const profitElement = document.getElementById('rewardTradeProfit');
        profitElement.textContent = `${(realizedProfitPct * 100).toFixed(2)}%`;
        profitElement.className = `trade-profit ${realizedProfitPct >= 0 ? 'positive' : 'negative'}`;

        // Populate reward components
        this.populateRewardComponents(components);

        // Show warnings if any
        this.showRewardWarnings(components);
    }

    populateRewardComponents(components) {
        const positiveContainer = document.getElementById('positiveComponents');
        const penaltyContainer = document.getElementById('penaltyComponents');
        
        positiveContainer.innerHTML = '';
        penaltyContainer.innerHTML = '';

        const positiveComponents = [];
        const penaltyComponents = [];

        // Categorize components
        Object.entries(components).forEach(([key, value]) => {
            if (key === 'killswitch' || key === 'killreason' || key === 'error') return;
            
            if (typeof value === 'number' && value !== 0) {
                const component = {
                    key: this.formatComponentLabel(key),
                    value: value,
                    isPositive: value > 0
                };
                
                if (value > 0) {
                    positiveComponents.push(component);
                } else {
                    penaltyComponents.push(component);
                }
            }
        });

        // Populate positive rewards
        if (positiveComponents.length > 0) {
            positiveComponents.forEach(comp => {
                const element = this.createComponentElement(comp.key, comp.value, 'positive');
                positiveContainer.appendChild(element);
            });
        } else {
            positiveContainer.innerHTML = '<div class="no-components">No positive rewards</div>';
        }

        // Populate penalties
        if (penaltyComponents.length > 0) {
            penaltyComponents.forEach(comp => {
                const element = this.createComponentElement(comp.key, Math.abs(comp.value), 'negative');
                penaltyContainer.appendChild(element);
            });
        } else {
            penaltyContainer.innerHTML = '<div class="no-components">No penalties</div>';
        }
    }

    createComponentElement(label, value, type) {
        const element = document.createElement('div');
        element.className = 'reward-component';
        
        element.innerHTML = `
            <span class="component-label">${label}</span>
            <span class="component-value ${type}">${type === 'positive' ? '+' : '-'}${value.toFixed(2)}</span>
        `;
        
        return element;
    }

    formatComponentLabel(key) {
        const labelMap = {
            'pnlpoints': 'Profit Points',
            'losspoints': 'Loss Points',
            'bandbonus': 'Band Bonus',
            'feespenalty': 'Fees & Slippage',
            'slviolation': 'Stop Loss Violation',
            'exposurepenalty': 'Excess Exposure',
            'leveragepenalty': 'Excess Leverage',
            'holdingdecay': 'Holding Time',
            'sharpebonus': 'Sharpe Bonus',
            'ddsoftpenalty': 'Drawdown Penalty',
            'consecutiveslpenalty': 'Consecutive SL'
        };
        
        return labelMap[key] || key.replace(/([A-Z])/g, ' $1').replace(/^./, str => str.toUpperCase());
    }

    showRewardWarnings(components) {
        const warningsContainer = document.getElementById('rewardWarnings');
        const warningText = document.getElementById('warningText');
        
        let warnings = [];
        
        if (components.killswitch) {
            warnings.push(`🛑 KILL SWITCH: ${components.killreason || 'Risk limit exceeded'}`);
        }
        
        if (components.error) {
            warnings.push(`⚠️ Calculation Error: ${components.error}`);
        }
        
        if (components.slviolation < 0) {
            warnings.push('🔻 Stop loss limit exceeded');
        }
        
        if (components.exposurepenalty > 0) {
            warnings.push('📊 Position size exceeds exposure limit');
        }
        
        if (warnings.length > 0) {
            warningText.textContent = warnings.join(' | ');
            warningsContainer.style.display = 'block';
        } else {
            warningsContainer.style.display = 'none';
        }
    }

    closeRewardModal() {
        const modal = document.getElementById('rewardModal');
        if (modal) {
            modal.classList.remove('show');
        }
        
        if (this.autoCloseTimer) {
            clearTimeout(this.autoCloseTimer);
            this.autoCloseTimer = null;
        }
    }

    addToRewardHistory(tradeData) {
        const historyItem = {
            ...tradeData,
            timestamp: new Date().toISOString(),
            displayTime: new Date().toLocaleTimeString()
        };
        
        this.rewardHistory.unshift(historyItem);
        
        // Keep only the most recent items
        if (this.rewardHistory.length > this.maxHistorySize) {
            this.rewardHistory = this.rewardHistory.slice(0, this.maxHistorySize);
        }
        
        this.saveRewardHistory();
        this.updateHistoryDisplay();
        this.updateHistoryStats();
    }

    updateHistoryDisplay() {
        const historyList = document.getElementById('rewardHistoryList');
        if (!historyList) return;
        
        if (this.rewardHistory.length === 0) {
            historyList.innerHTML = '<div class="no-rewards">No recent rewards</div>';
            return;
        }
        
        historyList.innerHTML = '';
        
        this.rewardHistory.slice(0, 20).forEach(item => {
            const historyElement = this.createHistoryElement(item);
            historyList.appendChild(historyElement);
        });
    }

    createHistoryElement(item) {
        const element = document.createElement('div');
        element.className = `reward-history-item ${item.reward >= 0 ? 'positive' : 'negative'}`;
        
        element.innerHTML = `
            <div class="reward-item-header">
                <span class="reward-item-symbol">${item.symbol}</span>
                <span class="reward-item-value ${item.reward >= 0 ? 'positive' : 'negative'}">
                    ${item.reward >= 0 ? '+' : ''}${item.reward.toFixed(2)}
                </span>
            </div>
            <div class="reward-item-details">
                <span>${item.displayTime}</span>
                <span>${(item.realizedProfitPct * 100).toFixed(1)}%</span>
            </div>
        `;
        
        // Click to show details
        element.addEventListener('click', () => {
            this.showRewardModal(item);
        });
        
        return element;
    }

    updateHistoryStats() {
        if (this.rewardHistory.length === 0) return;
        
        const avgReward = this.rewardHistory.reduce((sum, item) => sum + item.reward, 0) / this.rewardHistory.length;
        const totalPoints = this.rewardHistory.reduce((sum, item) => sum + item.reward, 0);
        
        const avgElement = document.getElementById('avgReward');
        const totalElement = document.getElementById('totalPoints');
        
        if (avgElement) avgElement.textContent = avgReward.toFixed(2);
        if (totalElement) totalElement.textContent = totalPoints.toFixed(0);
    }

    toggleRewardSidebar() {
        const sidebar = document.getElementById('rewardSidebar');
        if (!sidebar) return;
        
        this.sidebarOpen = !this.sidebarOpen;
        sidebar.classList.toggle('open', this.sidebarOpen);
    }

    saveRewardHistory() {
        try {
            localStorage.setItem('rewardHistory', JSON.stringify(this.rewardHistory));
        } catch (error) {
            console.warn('Failed to save reward history:', error);
        }
    }

    loadRewardHistory() {
        try {
            const saved = localStorage.getItem('rewardHistory');
            if (saved) {
                this.rewardHistory = JSON.parse(saved);
                // Ensure we don't exceed max size
                if (this.rewardHistory.length > this.maxHistorySize) {
                    this.rewardHistory = this.rewardHistory.slice(0, this.maxHistorySize);
                }
            }
        } catch (error) {
            console.warn('Failed to load reward history:', error);
            this.rewardHistory = [];
        }
    }

    // Public method to be called when a trade closes
    onTradeComplete(tradeData) {
        // Add to history
        this.addToRewardHistory(tradeData);
        
        // Show modal for significant rewards or penalties
        if (Math.abs(tradeData.reward) >= 5 || tradeData.components.killswitch) {
            this.showRewardModal(tradeData);
        }
        
        // Flash the sidebar toggle to indicate new reward
        this.flashSidebarToggle();
    }

    flashSidebarToggle() {
        const toggle = document.querySelector('.sidebar-toggle');
        if (toggle) {
            toggle.style.animation = 'pulse 0.6s ease-in-out';
            setTimeout(() => {
                toggle.style.animation = '';
            }, 600);
        }
    }
}

// Global functions for HTML onclick handlers
function closeRewardModal() {
    if (window.rewardManager) {
        window.rewardManager.closeRewardModal();
    }
}

function toggleRewardSidebar() {
    if (window.rewardManager) {
        window.rewardManager.toggleRewardSidebar();
    }
}

// Initialize reward manager when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.rewardManager = new RewardDisplayManager();
    
    // Example usage - remove in production
    // Simulate a trade completion after 5 seconds for testing
    setTimeout(() => {
        if (window.rewardManager && window.location.hostname === 'localhost') {
            const exampleTrade = {
                symbol: 'BTCUSDT',
                side: 'long',
                realizedProfitPct: 0.15,
                reward: 12.5,
                entryPrice: 50000,
                exitPrice: 57500,
                qty: 0.1,
                leverage: 2,
                components: {
                    pnlpoints: 5.0,
                    bandbonus: 3.0,
                    feespenalty: 2.5,
                    sharpebonus: 2.0,
                    killswitch: false
                }
            };
            // Uncomment next line for testing:
            // window.rewardManager.onTradeComplete(exampleTrade);
        }
    }, 5000);
});

// Clear Paper Trading History Function
function clearPaperTradingHistory(asset) {
    console.log(`Clear paper trading history called: ${asset}`);
    
    // Show confirmation dialog
    const confirmed = confirm(`Are you sure you want to clear all paper trading history for ${asset}?\n\nThis will:\n• Reset balance to starting amount\n• Clear all active positions\n• Clear all trade history\n• Reset all statistics\n\nThis action cannot be undone.`);
    
    if (!confirmed) {
        return;
    }
    
    // Show loading state
    const button = event?.target;
    if (button) {
        button.disabled = true;
        button.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Clearing...';
    }
    
    // Make API call to clear history
    fetch(`/asset/${asset}/clear-history`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        }
    })
    .then(response => response.json())
    .then(data => {
        if (data.status === 'cleared') {
            // Success - refresh the dashboard
            if (dashboard) {
                dashboard.showSystemMessage(`${asset.toUpperCase()} paper trading history cleared successfully!`, 'success');
                dashboard.loadAllData(); // Refresh all data
            }
            console.log(`Paper trading history cleared for ${asset}:`, data);
        } else {
            throw new Error(data.message || 'Failed to clear history');
        }
    })
    .catch(error => {
        console.error(`Error clearing ${asset} history:`, error);
        if (dashboard) {
            dashboard.showSystemMessage(`Failed to clear ${asset} history: ${error.message}`, 'error');
        } else {
            alert(`Error clearing ${asset} history: ${error.message}`);
        }
    })
    .finally(() => {
        // Restore button state
        if (button) {
            button.disabled = false;
            button.innerHTML = '<i class="fas fa-trash"></i> Clear';
        }
    });
}