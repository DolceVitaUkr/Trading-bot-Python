class MultiAssetDashboard {
    constructor() {
        this.baseUrl = '';
        this.updateInterval = 3000; // 3 seconds
        this.updateTimer = null;
        this.assets = ['crypto', 'futures', 'forex', 'forex_options'];
        this.charts = {};
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
            } catch (error) {
                console.error('Periodic update failed:', error);
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
            this.updateElement(`${asset}-${type}-balance-value`, '$0.00');
            this.updateElement(`${asset}-${type}-change`, 'No data');
            return;
        }

        const balance = walletData.balance || 0;
        const pnl = walletData.pnl || 0;
        const pnlPercent = walletData.pnl_percent || 0;

        this.updateElement(`${asset}-${type}-balance-value`, this.formatCurrency(balance));
        
        const changeElement = document.getElementById(`${asset}-${type}-change`);
        if (changeElement) {
            const changeText = `${pnl >= 0 ? '+' : ''}${this.formatCurrency(pnl)} (${pnlPercent >= 0 ? '+' : ''}${pnlPercent.toFixed(2)}%)`;
            changeElement.textContent = changeText;
            changeElement.className = `balance-change ${pnl >= 0 ? 'positive' : 'negative'}`;
        }
        
        // Also update the simple balance elements for compatibility
        this.updateElement(`${asset}-${type}-balance`, this.formatCurrency(balance));
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
        // Update position count in metrics for both paper and live
        const modes = ['paper', 'live'];
        
        modes.forEach(mode => {
            const positionElement = document.getElementById(`${asset}-${mode}-positions`);
            const strategiesElement = document.getElementById(`${asset}-${mode}-strategies`);
            const winRateElement = document.getElementById(`${asset}-${mode}-winrate`);
            const dailyPnlElement = document.getElementById(`${asset}-${mode}-dailypnl`);
            const positionsListElement = document.getElementById(`${asset}-${mode}-positions-list`);
            
            const positions = positionsData.positions || [];
            
            // Update metrics
            if (positionElement) positionElement.textContent = positions.length;
            if (strategiesElement) strategiesElement.textContent = positionsData.active_strategies || 0;
            if (winRateElement) winRateElement.textContent = `${(positionsData.win_rate || 0).toFixed(1)}%`;
            
            if (dailyPnlElement) {
                const dailyPnl = positionsData.daily_pnl || 0;
                dailyPnlElement.textContent = this.formatCurrency(dailyPnl);
                dailyPnlElement.className = `metric-value ${dailyPnl >= 0 ? 'positive' : 'negative'}`;
            }
            
            // Update positions list
            if (positionsListElement) {
                this.updatePositionsList(positionsListElement, positions);
            }
        });
    }
    
    updatePositionsList(container, positions) {
        if (!positions || positions.length === 0) {
            container.innerHTML = '<div class="no-positions">No active positions</div>';
            return;
        }
        
        const positionsHtml = positions.map(position => {
            const pnlClass = position.pnl >= 0 ? 'positive' : 'negative';
            const pnlPercent = position.entry_price ? ((position.current_price - position.entry_price) / position.entry_price * 100) : 0;
            
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
                                ${position.pnl >= 0 ? '+' : ''}${this.formatCurrency(position.pnl || 0)}
                            </span>
                        </div>
                        <div class="position-detail">
                            <span class="position-label">P&L %</span>
                            <span class="position-value position-pnl ${pnlClass}">
                                ${pnlPercent >= 0 ? '+' : ''}${pnlPercent.toFixed(2)}%
                            </span>
                        </div>
                    </div>
                </div>
            `;
        }).join('');
        
        container.innerHTML = positionsHtml;
    }

    updateStrategyStatus(asset, strategyData) {
        const statusElement = document.getElementById(`${asset}-strategy-status`);
        if (!statusElement) return;

        if (!strategyData.strategies || strategyData.strategies.length === 0) {
            statusElement.textContent = 'No strategies active';
            return;
        }

        const activeStrategies = strategyData.strategies.filter(s => s.active).length;
        const testingStrategies = strategyData.strategies.filter(s => s.status === 'testing').length;
        const approvedStrategies = strategyData.strategies.filter(s => s.status === 'approved').length;

        let statusText = '';
        if (testingStrategies > 0) {
            statusText += `Testing ${testingStrategies} strategies`;
        }
        if (approvedStrategies > 0) {
            if (statusText) statusText += ', ';
            statusText += `${approvedStrategies} approved for live`;
        }
        if (!statusText) {
            statusText = `${activeStrategies} strategies active`;
        }

        statusElement.textContent = statusText;
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