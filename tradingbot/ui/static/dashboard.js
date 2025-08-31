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
        
        // Bind toggle switches for each asset
        this.assets.forEach(asset => {
            // Paper trading toggle
            const paperToggle = document.getElementById(`${asset}-paper-toggle`);
            if (paperToggle) {
                paperToggle.addEventListener('change', (e) => this.handlePaperToggle(asset, e.target.checked));
            }
            
            // Live trading toggle
            const liveToggle = document.getElementById(`${asset}-live-toggle`);
            if (liveToggle) {
                liveToggle.addEventListener('change', (e) => this.handleLiveToggle(asset, e.target.checked));
            }
        });
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
                    this.updateAllAssets(),
                    this.updateRecentActivities()
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
            this.updateElement('totalPaperWallet', this.formatCurrency(data.total_paper_wallet || 0));
            this.updateElement('totalLiveWallet', this.formatCurrency(data.total_live_wallet || 0));
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
    
    async updateRecentActivities() {
        try {
            const response = await fetch(`${this.baseUrl}/activity/recent`);
            const data = await response.json();
            
            console.log('Activity data received:', data);  // Debug log
            
            const activityList = document.getElementById('activityList');
            if (!activityList) {
                console.error('Activity list element not found');
                return;
            }
            
            if (data.activities && data.activities.length > 0) {
                // Clear existing content
                activityList.innerHTML = '';
                
                // Add each activity
                data.activities.slice(0, 20).forEach(activity => {
                    const activityItem = document.createElement('div');
                    activityItem.className = `activity-item ${activity.type || 'info'}`;
                    
                    const time = new Date(activity.timestamp).toLocaleTimeString('en-US', {
                        hour: '2-digit',
                        minute: '2-digit',
                        hour12: false
                    });
                    
                    activityItem.innerHTML = `
                        <span class="activity-time">${time}</span>
                        <span class="activity-source">[${activity.source || 'SYSTEM'}]</span>
                        <span class="activity-message">${activity.message}</span>
                    `;
                    
                    activityList.appendChild(activityItem);
                });
            } else {
                console.log('No activities found');
                if (activityList.children.length === 0) {
                    activityList.innerHTML = '<div class="no-data">No recent activity</div>';
                }
            }
        } catch (error) {
            console.error('Failed to update activities:', error);
        }
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

            // Update positions - now handles both positions table and activity log
            this.updatePositions(asset, positions);
            this.updatePositionsTable(asset, 'paper', positions.paper?.positions || []);
            this.updatePositionsTable(asset, 'live', positions.live?.positions || []);

            // Update trade history
            if (positions.paper?.positions?.length > 0 || positions.live?.positions?.length > 0) {
                // Load detailed trades for history table
                this.loadTradeHistory(asset, 'paper');
                this.loadTradeHistory(asset, 'live');
            }

            // Update strategy status
            this.updateStrategyStatus(asset, strategies);

            // Update control buttons based on status
            this.updateAssetControls(asset, status);
            
            // Update training status display
            this.updateTrainingStatus(asset, status);
            
            // Update asset-specific activity logs
            this.updateAssetActivityLogs(asset);

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
    
    async updateAssetActivityLogs(asset) {
        try {
            // Get recent activities from the global activity feed filtered by asset
            const response = await fetch(`${this.baseUrl}/activity/recent`);
            const data = await response.json();
            
            if (data.activities && data.activities.length > 0) {
                // Filter activities for this specific asset
                const assetActivities = data.activities.filter(activity => {
                    return activity.source && activity.source.toLowerCase() === asset.toLowerCase();
                });
                
                // Update both paper and live activity logs
                this.updateActivityLog(asset, 'paper', assetActivities.filter(a => a.message && a.message.toLowerCase().includes('paper')));
                this.updateActivityLog(asset, 'live', assetActivities.filter(a => a.message && a.message.toLowerCase().includes('live')));
            }
        } catch (error) {
            console.error(`Failed to update activity logs for ${asset}:`, error);
        }
    }
    
    updateActivityLog(asset, mode, activities) {
        const listId = `${asset}-${mode}-activity-list`;
        const list = document.getElementById(listId);
        if (!list) return;
        
        // Clear current content
        list.innerHTML = '';
        
        if (!activities || activities.length === 0) {
            list.innerHTML = '<div class="no-activity" style="text-align: center; color: #6c757d; padding: 20px; font-size: 12px;">No activity yet</div>';
            return;
        }
        
        // Add each activity (most recent first, limited to 10)
        activities.slice(0, 10).forEach(activity => {
            const item = document.createElement('div');
            item.className = `activity-item ${activity.type || 'info'}`;
            item.style.cssText = 'padding: 8px 12px; border-bottom: 1px solid #2d3748; font-size: 12px;';
            
            const time = activity.timestamp ? new Date(activity.timestamp).toLocaleTimeString('en-US', { 
                hour: '2-digit', 
                minute: '2-digit' 
            }) : '';
            
            const icon = activity.type === 'trade' ? '📊' : 
                        activity.type === 'error' ? '❌' : 
                        activity.type === 'warning' ? '⚠️' :
                        activity.type === 'success' ? '✅' : 'ℹ️';
            
            item.innerHTML = `
                <div style="display: flex; align-items: center; gap: 8px;">
                    <span style="font-size: 14px;">${icon}</span>
                    <span style="color: #a0aec0; font-size: 11px;">${time}</span>
                    <span style="color: #e2e8f0; flex: 1;">${activity.message || 'No message'}</span>
                </div>
            `;
            
            list.appendChild(item);
        });
    }
    
    async loadTradeHistory(asset, mode) {
        try {
            const response = await fetch(`${this.baseUrl}/asset/${asset}/positions`);
            const data = await response.json();
            
            const modeData = mode === 'paper' ? data.paper : data.live;
            const positions = modeData?.positions || [];
            
            // Update the trade history table
            const tbody = document.getElementById(`${asset}-${mode}-history-tbody`);
            if (!tbody) return;
            
            if (positions.length === 0) {
                tbody.innerHTML = '<tr><td colspan="10" style="text-align: center; padding: 20px; color: #6c757d;">No completed trades yet</td></tr>';
                return;
            }
            
            tbody.innerHTML = '';
            
            // Add closed positions to history
            positions.filter(p => p.status === 'closed').forEach(position => {
                const row = document.createElement('tr');
                const pnlClass = position.pnl >= 0 ? 'positive' : 'negative';
                
                row.innerHTML = `
                    <td>${mode.toUpperCase()}</td>
                    <td>${position.side || 'N/A'}</td>
                    <td>${this.formatTimestamp(position.entry_time)}</td>
                    <td>${this.formatCurrency(position.size_usd || 0)}</td>
                    <td>${this.formatPrice(position.entry_price || 0)}</td>
                    <td>${this.formatTimestamp(position.exit_time)}</td>
                    <td>${this.formatPrice(position.exit_price || 0)}</td>
                    <td class="${pnlClass}">${this.formatCurrency(position.pnl || 0)}</td>
                    <td class="${pnlClass}">${position.pnl_pct?.toFixed(2) || '0.00'}%</td>
                    <td>${this.formatCurrency(position.balance_after || 0)}</td>
                `;
                
                tbody.appendChild(row);
            });
            
        } catch (error) {
            console.error(`Failed to load trade history for ${asset} ${mode}:`, error);
        }
    }
    
    formatTimestamp(timestamp) {
        if (!timestamp) return 'N/A';
        const date = new Date(timestamp);
        return date.toLocaleString('en-US', { 
            month: 'short', 
            day: 'numeric',
            hour: '2-digit',
            minute: '2-digit'
        });
    }

    updateWallet(asset, type, walletData) {
        if (!walletData) {
            this.updateElement(`${asset}-${type}-balance`, '$0.00');
            this.updateElement(`${asset}-${type}-change`, 'No data');
            return;
        }

        const balance = walletData.balance || 0;
        const pnl = walletData.pnl || 0;
        const pnlPercent = walletData.pnl_percent || 0;
        
        // Update balance display
        this.updateElement(`${asset}-${type}-balance`, this.formatCurrency(balance));
        
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
        
        console.log(`Updating chart for ${chartKey}:`, walletData);
        console.log(`Chart exists: ${!!chart}`);
        console.log(`Wallet data exists: ${!!walletData}`);
        console.log(`History exists: ${!!walletData?.history}`);
        console.log(`History length: ${walletData?.history?.length || 0}`);
        
        if (!chart || !walletData || !walletData.history) {
            console.log(`Chart update skipped for ${chartKey}: chart=${!!chart}, walletData=${!!walletData}, history=${walletData?.history?.length || 0}`);
            return;
        }

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
        console.log(`Updating positions for ${asset}:`, positionsData);
        
        // Handle the nested position data structure - API returns paper_trading/live_trading
        const paperData = positionsData.paper_trading || positionsData.paper || {};
        const liveData = positionsData.live_trading || positionsData.live || {};
        
        const paperPositions = paperData.positions || [];
        const livePositions = liveData.positions || [];
        
        console.log(`Paper positions count: ${paperPositions.length}`, paperPositions);
        console.log(`Live positions count: ${livePositions.length}`, livePositions);
        
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
        // First check if we're dealing with the new table format
        const isTable = container.id && container.id.includes('-positions-table');
        const tbody = isTable ? document.getElementById(container.id.replace('-table', '-tbody')) : null;
        
        if (isTable && tbody) {
            // New table format
            this.updatePositionsTable(container.id.split('-')[0], container.id.split('-')[1], positions);
            return;
        }
        
        // Old format handling (for backwards compatibility)
        const section = container.closest('.positions-section');
        
        if (!positions || positions.length === 0) {
            if (section) section.style.display = 'none';
            container.innerHTML = '<div class="no-positions" style="color: #6c757d; font-size: 12px;">No active positions</div>';
            return;
        }
        
        if (section) section.style.display = 'block';
        
        container.innerHTML = '';
        
        positions.forEach(position => {
            const pnl = position.pnl || 0;
            const pnlPct = position.pnl_pct || 0;
            const pnlClass = pnl >= 0 ? 'positive' : 'negative';
            const sideClass = position.side ? position.side.toLowerCase() : '';
            
            const positionDiv = document.createElement('div');
            positionDiv.style.cssText = 'padding: 8px; border-bottom: 1px solid #2d3748; font-size: 12px;';
            positionDiv.innerHTML = `
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <span style="font-weight: 600; color: #e2e8f0;">${position.symbol || 'N/A'}</span>
                        <span class="position-side ${sideClass}" style="margin-left: 8px; font-size: 11px; text-transform: uppercase;">${position.side || 'N/A'}</span>
                    </div>
                    <div style="text-align: right;">
                        <div class="position-pnl ${pnlClass}" style="font-weight: 600;">
                            ${pnl >= 0 ? '+' : ''}${this.formatCurrency(pnl)} (${pnlPct >= 0 ? '+' : ''}${pnlPct.toFixed(2)}%)
                        </div>
                        <div style="font-size: 11px; color: #8892b0;">
                            Entry: ${this.formatPrice(position.entry_price || 0)}
                        </div>
                    </div>
                </div>
            `;
            
            container.appendChild(positionDiv);
        });
    }
    
    // New function for updating positions table
    updatePositionsTable(asset, mode, positions) {
        const tbody = document.getElementById(`${asset}-${mode}-positions-tbody`);
        if (!tbody) return;
        
        if (!positions || positions.length === 0) {
            tbody.innerHTML = '<tr><td colspan="7" style="text-align: center; padding: 20px; color: #6c757d;">No active positions</td></tr>';
            const section = document.getElementById(`${asset}-${mode}-positions-section`);
            if (section) section.style.display = 'block';
            return;
        }
        
        const section = document.getElementById(`${asset}-${mode}-positions-section`);
        if (section) section.style.display = 'block';
        
        tbody.innerHTML = '';
        
        positions.forEach(position => {
            const row = document.createElement('tr');
            const pnl = position.pnl || 0;
            const pnlPct = position.pnl_pct || 0;
            const pnlClass = pnl >= 0 ? 'positive' : 'negative';
            const sideClass = position.side ? position.side.toLowerCase() : '';
            
            row.innerHTML = `
                <td style="font-weight: 600;">${position.symbol}</td>
                <td class="${sideClass}">${position.side}</td>
                <td>$${(position.size * position.entry_price).toFixed(2)}</td>
                <td>$${position.entry_price.toFixed(position.entry_price < 1 ? 6 : 2)}</td>
                <td>$${position.current_price.toFixed(position.current_price < 1 ? 6 : 2)}</td>
                <td class="${pnlClass}" style="font-weight: 600;">
                    ${pnl >= 0 ? '+' : ''}$${Math.abs(pnl).toFixed(2)} (${pnlPct >= 0 ? '+' : ''}${pnlPct.toFixed(2)}%)
                </td>
                <td style="font-size: 11px;">
                    TP: ${position.take_profit ? '$' + position.take_profit.toFixed(2) : '-'}<br>
                    SL: ${position.stop_loss ? '$' + position.stop_loss.toFixed(2) : '-'}
                </td>
            `;
            
            // Store data for sorting
            row.dataset.symbol = position.symbol;
            row.dataset.side = position.side;
            row.dataset.size = position.size * position.entry_price;
            row.dataset.pnl = pnl;
            
            tbody.appendChild(row);
        });
        
        // Log activity for positions
        this.updateAssetActivity(asset, mode, {
            type: 'position',
            message: `${positions.length} active position(s)`,
            timestamp: new Date().toISOString()
        });
    }

    updateActivityLog(asset, mode, activities) {
        const listId = `${asset}-${mode}-activity-list`;
        const list = document.getElementById(listId);
        if (!list) return;
        
        // Clear current content
        list.innerHTML = '';
        
        if (!activities || activities.length === 0) {
            list.innerHTML = '<div class="no-activity" style="text-align: center; color: #6c757d; padding: 20px; font-size: 12px;">No activity yet</div>';
            return;
        }
        
        // Add each activity
        activities.forEach(activity => {
            const item = document.createElement('div');
            item.className = `activity-item ${activity.type}`;
            item.style.cssText = 'padding: 8px 12px; border-bottom: 1px solid #2d3748; font-size: 12px;';
            
            const time = new Date(activity.timestamp).toLocaleTimeString();
            const icon = activity.type === 'trade' ? '📊' : 
                        activity.type === 'error' ? '⚠️' : 
                        activity.type === 'position' ? '📈' : '📝';
            
            item.innerHTML = `
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span>${icon} ${activity.message}</span>
                    <span style="color: #6c757d; font-size: 11px;">${time}</span>
                </div>
            `;
            
            list.appendChild(item);
        });
    }

    updateTradeHistory(asset, mode, trades) {
        const tbodyId = `${asset}-${mode}-history-tbody`;
        const tbody = document.getElementById(tbodyId);
        if (!tbody) return;
        
        // Clear current content
        tbody.innerHTML = '';
        
        if (!trades || trades.length === 0) {
            tbody.innerHTML = '<tr><td colspan="10" style="text-align: center; padding: 20px; color: #6c757d;">No completed trades yet</td></tr>';
            return;
        }
        
        // Add each trade
        trades.forEach(trade => {
            const row = document.createElement('tr');
            const pnlClass = trade.pnl >= 0 ? 'positive' : 'negative';
            
            row.innerHTML = `
                <td style="padding: 8px; color: #e2e8f0;">${mode.toUpperCase()}</td>
                <td style="padding: 8px; color: #e2e8f0;">${trade.side}</td>
                <td style="padding: 8px; color: #e2e8f0;">${new Date(trade.open_time).toLocaleString()}</td>
                <td style="padding: 8px; color: #e2e8f0;">$${trade.size_usd.toFixed(2)}</td>
                <td style="padding: 8px; color: #e2e8f0;">$${trade.open_price.toFixed(4)}</td>
                <td style="padding: 8px; color: #e2e8f0;">${new Date(trade.close_time).toLocaleString()}</td>
                <td style="padding: 8px; color: #e2e8f0;">$${trade.close_price.toFixed(4)}</td>
                <td style="padding: 8px;" class="${pnlClass}">$${trade.pnl.toFixed(2)}</td>
                <td style="padding: 8px;" class="${pnlClass}">${trade.pnl_pct.toFixed(2)}%</td>
                <td style="padding: 8px; color: #e2e8f0;">$${trade.balance.toFixed(2)}</td>
            `;
            
            tbody.appendChild(row);
        });
    }

    
    // Function to update asset activity log
    updateAssetActivity(asset, mode, activity) {
        const activityList = document.getElementById(`${asset}-${mode}-activity-list`);
        if (!activityList) return;
        
        // Remove "no activity" message if present
        const noActivity = activityList.querySelector('.no-activity');
        if (noActivity) {
            noActivity.remove();
        }
        
        // Create activity item
        const activityItem = document.createElement('div');
        activityItem.className = `activity-item ${activity.type}`;
        activityItem.style.cssText = 'padding: 6px 10px; border-left: 3px solid #4a5568; margin-bottom: 8px; background: #2d3748; border-radius: 4px; font-size: 12px; color: #e2e8f0;';
        
        if (activity.type === 'trade') activityItem.style.borderLeftColor = '#48bb78';
        else if (activity.type === 'error') activityItem.style.borderLeftColor = '#f56565';
        else if (activity.type === 'position') activityItem.style.borderLeftColor = '#4299e1';
        
        const timeDiv = document.createElement('div');
        timeDiv.className = 'activity-time';
        timeDiv.style.cssText = 'font-size: 11px; color: #718096; margin-bottom: 2px;';
        timeDiv.textContent = new Date(activity.timestamp).toLocaleTimeString();
        
        const messageDiv = document.createElement('div');
        messageDiv.className = 'activity-message';
        messageDiv.innerHTML = `<strong>[${mode.toUpperCase()}]</strong> ${activity.message}`;
        
        activityItem.appendChild(timeDiv);
        activityItem.appendChild(messageDiv);
        
        // Add to top of list
        activityList.insertBefore(activityItem, activityList.firstChild);
        
        // Keep only last 50 activities
        while (activityList.children.length > 50) {
            activityList.removeChild(activityList.lastChild);
        }
    }
    
    // Function to update history table
    updateHistoryTable(asset, mode, trades) {
        const tbody = document.getElementById(`${asset}-${mode}-history-tbody`);
        if (!tbody) return;
        
        if (!trades || trades.length === 0) {
            tbody.innerHTML = '<tr><td colspan="10" style="text-align: center; padding: 20px; color: #6c757d;">No completed trades yet</td></tr>';
            return;
        }
        
        tbody.innerHTML = '';
        
        trades.forEach(trade => {
            const row = document.createElement('tr');
            const pnl = trade.pnl || 0;
            const pnlPct = trade.pnl_pct || 0;
            const pnlClass = pnl >= 0 ? 'positive' : 'negative';
            
            row.innerHTML = `
                <td><span class="badge ${mode}" style="padding: 2px 6px; border-radius: 3px; font-size: 10px; background: ${mode === 'paper' ? '#3b82f6' : '#f59e0b'}; color: white;">${mode.toUpperCase()}</span></td>
                <td class="${trade.side.toLowerCase()}">${trade.side}</td>
                <td>${new Date(trade.open_time).toLocaleString()}</td>
                <td>$${trade.size_usd.toFixed(2)}</td>
                <td>$${trade.open_price.toFixed(trade.open_price < 1 ? 6 : 2)}</td>
                <td>${new Date(trade.close_time).toLocaleString()}</td>
                <td>$${trade.close_price.toFixed(trade.close_price < 1 ? 6 : 2)}</td>
                <td class="${pnlClass}" style="font-weight: 600;">${pnl >= 0 ? '+' : ''}$${Math.abs(pnl).toFixed(2)}</td>
                <td class="${pnlClass}">${pnlPct >= 0 ? '+' : ''}${pnlPct.toFixed(2)}%</td>
                <td>$${trade.balance_after.toFixed(2)}</td>
            `;
            
            tbody.appendChild(row);
        });
    }
    
    formatPrice(price) {
        if (!price || price === 0) return '-';
        return price < 1 ? price.toFixed(6) : price.toFixed(2);
    }
    
    calculateDurationFromNow(timestamp) {
        if (!timestamp) return '-';
        const start = new Date(timestamp);
        const now = new Date();
        const diff = now - start;
        
        const hours = Math.floor(diff / (1000 * 60 * 60));
        const minutes = Math.floor((diff % (1000 * 60 * 60)) / (1000 * 60));
        
        if (hours > 0) {
            return `${hours}h ${minutes}m`;
        }
        return `${minutes}m`;
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
        const paperToggle = document.getElementById(`${asset}-paper-toggle`);
        const liveToggle = document.getElementById(`${asset}-live-toggle`);
        const killBtn = document.getElementById(`${asset}-kill`);
        const paperStatus = document.getElementById(`${asset}-paper-status`);
        const liveStatus = document.getElementById(`${asset}-live-status`);

        // Update paper trading toggle and status
        if (paperToggle) {
            paperToggle.checked = status.paper_trading_active || false;
            this.updateTradingStatus(paperStatus, status.paper_trading_active ? 'learning' : 'idle', 
                                   status.paper_trading_active ? 'Learning' : 'Idle');
        }

        // Update live trading toggle and status
        if (liveToggle) {
            liveToggle.checked = status.live_trading_active || false;
            liveToggle.disabled = !status.live_trading_approved;
            this.updateTradingStatus(liveStatus, status.live_trading_active ? 'active' : 'idle', 
                                   status.live_trading_active ? 'Live Trading' : 'Idle');
        }

        // Update kill switch
        if (killBtn) {
            if (status.kill_switch_active) {
                killBtn.textContent = 'KILLED';
                killBtn.className = 'btn btn-kill disabled';
                killBtn.disabled = true;
                this.updateTradingStatus(paperStatus, 'error', 'Killed');
                this.updateTradingStatus(liveStatus, 'error', 'Killed');
                
                // Uncheck toggles when killed
                if (paperToggle) paperToggle.checked = false;
                if (liveToggle) liveToggle.checked = false;
            } else {
                killBtn.textContent = 'KILL';
                killBtn.className = 'btn btn-kill';
                killBtn.disabled = false;
            }
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
            statusElement.className = status === 'online' ? 'status-connected' : 'status-offline';
            statusElement.textContent = text || (status === 'online' ? 'Online' : 'Offline');
        }
    }
    
    updateBybitStatus(status) {
        const statusElement = document.getElementById('bybit-status');
        if (statusElement) {
            const isConnected = status === 'connected';
            statusElement.className = isConnected ? 'status-connected' : 'status-offline';
            
            const statusText = status === 'connected' ? 'Connected' :
                              status === 'auth_failed' ? 'Auth Failed' : 'Offline';
            statusElement.textContent = statusText;
        }
    }
    
    updateIbkrStatus(status) {
        const statusElement = document.getElementById('ibkr-status');
        if (statusElement) {
            const isConnected = status === 'connected';
            statusElement.className = isConnected ? 'status-connected' : 'status-offline';
            statusElement.textContent = isConnected ? 'Connected' : 'Offline';
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
    
    addActivityLog(asset, message, type = 'info') {
        const activityContainer = document.getElementById(`${asset}-activity`);
        if (!activityContainer) return;
        
        const now = new Date();
        const timeStr = now.toLocaleTimeString('en-US', { 
            hour: '2-digit', 
            minute: '2-digit',
            hour12: false 
        });
        
        const activityLine = document.createElement('div');
        activityLine.className = `activity-line ${type}`;
        activityLine.innerHTML = `
            <span class="activity-time">${timeStr}</span>
            <span class="activity-text">${message}</span>
        `;
        
        // Add to top and keep only last 4 entries
        activityContainer.insertBefore(activityLine, activityContainer.firstChild);
        while (activityContainer.children.length > 4) {
            activityContainer.removeChild(activityContainer.lastChild);
        }
    }
    
    logActivity(asset, message, type = 'info') {
        // Alias for addActivityLog - normalize asset name
        this.addActivityLog(asset.toLowerCase(), message, type);
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
    
    async handlePaperToggle(asset, isEnabled) {
        const action = isEnabled ? 'start' : 'stop';
        const assetUpper = asset.toUpperCase();
        
        console.log(`${action}ing paper trading for ${assetUpper}...`);
        this.logActivity(assetUpper, `${action === 'start' ? 'Starting' : 'Stopping'} paper trading...`);
        
        try {
            const response = await fetch(`${this.baseUrl}/asset/${asset}/${action}/paper`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' }
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const data = await response.json();
            this.logActivity(assetUpper, data.message || `Paper trading ${action}ed`, 'success');
            
            // Update asset status
            await this.updateAsset(asset);

        } catch (error) {
            console.error(`Failed to ${action} paper trading for ${asset}:`, error);
            this.logActivity(assetUpper, `Failed to ${action} paper trading`, 'error');
            
            // Reset toggle on failure
            const toggle = document.getElementById(`${asset}-paper-toggle`);
            if (toggle) toggle.checked = !isEnabled;
        }
    }
    
    async handleLiveToggle(asset, isEnabled) {
        if (isEnabled) {
            // Double confirmation for live trading
            const confirmed = confirm(`⚠️ LIVE TRADING CONFIRMATION ⚠️\n\nYou are about to enable LIVE TRADING for ${asset.toUpperCase()}.\n\nThis will use REAL MONEY and execute REAL trades.\n\nAre you absolutely sure?`);
            
            if (!confirmed) {
                // Reset toggle
                const toggle = document.getElementById(`${asset}-live-toggle`);
                if (toggle) toggle.checked = false;
                return;
            }
            
            const doubleConfirmed = confirm(`⚠️ FINAL CONFIRMATION ⚠️\n\nLAST CHANCE: This will enable LIVE TRADING with REAL MONEY for ${asset.toUpperCase()}.\n\nClick OK to proceed with LIVE trading.`);
            
            if (!doubleConfirmed) {
                // Reset toggle
                const toggle = document.getElementById(`${asset}-live-toggle`);
                if (toggle) toggle.checked = false;
                return;
            }
        }
        
        const action = isEnabled ? 'start' : 'stop';
        const assetUpper = asset.toUpperCase();
        
        console.log(`${action}ing live trading for ${assetUpper}...`);
        this.logActivity(assetUpper, `${action === 'start' ? 'Starting' : 'Stopping'} live trading...`, 'warning');
        
        try {
            const response = await fetch(`${this.baseUrl}/asset/${asset}/${action}/live`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' }
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const data = await response.json();
            this.logActivity(assetUpper, data.message || `Live trading ${action}ed`, isEnabled ? 'warning' : 'success');
            
            // Update asset status
            await this.updateAsset(asset);

        } catch (error) {
            console.error(`Failed to ${action} live trading for ${asset}:`, error);
            this.logActivity(assetUpper, `Failed to ${action} live trading`, 'error');
            
            // Reset toggle on failure
            const toggle = document.getElementById(`${asset}-live-toggle`);
            if (toggle) toggle.checked = !isEnabled;
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
    
    // Add sorting functions for positions table
    sortPositions(asset, mode, column) {
        const tbody = document.getElementById(`${asset}-${mode}-positions-tbody`);
        if (!tbody) return;
        
        const rows = Array.from(tbody.querySelectorAll('tr'));
        if (rows.length === 0 || rows[0].querySelector('td[colspan]')) return;
        
        rows.sort((a, b) => {
            const aVal = a.dataset[column];
            const bVal = b.dataset[column];
            
            if (column === 'symbol' || column === 'side') {
                return aVal.localeCompare(bVal);
            } else {
                return parseFloat(bVal) - parseFloat(aVal);
            }
        });
        
        tbody.innerHTML = '';
        rows.forEach(row => tbody.appendChild(row));
    }
    
    // Setup filters and event handlers
    setupFilters() {
        // Position search
        document.querySelectorAll('[id$="-positions-search"]').forEach(input => {
            input.addEventListener('input', (e) => {
                const parts = e.target.id.split('-');
                const asset = parts[0];
                const mode = parts[1];
                const searchTerm = e.target.value.toLowerCase();
                const tbody = document.getElementById(`${asset}-${mode}-positions-tbody`);
                
                tbody.querySelectorAll('tr').forEach(row => {
                    const symbol = row.dataset.symbol;
                    if (symbol && symbol.toLowerCase().includes(searchTerm)) {
                        row.style.display = '';
                    } else if (!row.querySelector('td[colspan]')) {
                        row.style.display = 'none';
                    }
                });
            });
        });
        
        // Position sort dropdown
        document.querySelectorAll('[id$="-positions-sort"]').forEach(select => {
            select.addEventListener('change', (e) => {
                const parts = e.target.id.split('-');
                const asset = parts[0];
                const mode = parts[1];
                const sortBy = e.target.value;
                
                let column = 'symbol';
                if (sortBy === 'pnl-desc' || sortBy === 'pnl-asc') column = 'pnl';
                else if (sortBy === 'size-desc') column = 'size';
                
                this.sortPositions(asset, mode, column);
            });
        });
        
        // Activity filter
        document.querySelectorAll('[id$="-activity-filter"]').forEach(select => {
            select.addEventListener('change', (e) => {
                const parts = e.target.id.split('-');
                const asset = parts[0];
                const mode = parts[1];
                const filterType = e.target.value;
                const activityList = document.getElementById(`${asset}-${mode}-activity-list`);
                
                activityList.querySelectorAll('.activity-item').forEach(item => {
                    if (filterType === 'all' || item.classList.contains(filterType)) {
                        item.style.display = '';
                    } else {
                        item.style.display = 'none';
                    }
                });
            });
        });
        
        // History search
        document.querySelectorAll('[id$="-history-search"]').forEach(input => {
            input.addEventListener('input', (e) => {
                const parts = e.target.id.split('-');
                const asset = parts[0];
                const mode = parts[1];
                const searchTerm = e.target.value.toLowerCase();
                const tbody = document.getElementById(`${asset}-${mode}-history-tbody`);
                
                tbody.querySelectorAll('tr').forEach(row => {
                    const text = row.textContent.toLowerCase();
                    if (text.includes(searchTerm)) {
                        row.style.display = '';
                    } else if (!row.querySelector('td[colspan]')) {
                        row.style.display = 'none';
                    }
                });
            });
        });
        
        // History sort dropdown
        document.querySelectorAll('[id$="-history-sort"]').forEach(select => {
            select.addEventListener('change', (e) => {
                const parts = e.target.id.split('-');
                const asset = parts[0];
                const mode = parts[1];
                const sortBy = e.target.value;
                
                // Implement history sorting based on the selected option
                this.sortHistoryTable(asset, mode, sortBy);
            });
        });
    }
    
    // Sort history table
    sortHistoryTable(asset, mode, sortBy) {
        const tbody = document.getElementById(`${asset}-${mode}-history-tbody`);
        if (!tbody) return;
        
        const rows = Array.from(tbody.querySelectorAll('tr'));
        if (rows.length === 0 || rows[0].querySelector('td[colspan]')) return;
        
        rows.sort((a, b) => {
            switch(sortBy) {
                case 'time-desc':
                    return new Date(b.cells[2].textContent) - new Date(a.cells[2].textContent);
                case 'time-asc':
                    return new Date(a.cells[2].textContent) - new Date(b.cells[2].textContent);
                case 'pnl-desc':
                    return parseFloat(b.cells[7].textContent.replace(/[$,+]/g, '')) - parseFloat(a.cells[7].textContent.replace(/[$,+]/g, ''));
                case 'pnl-asc':
                    return parseFloat(a.cells[7].textContent.replace(/[$,+]/g, '')) - parseFloat(b.cells[7].textContent.replace(/[$,+]/g, ''));
                default:
                    return 0;
            }
        });
        
        tbody.innerHTML = '';
        rows.forEach(row => tbody.appendChild(row));
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

// Handle toggle switch changes
async function handleTradingToggle(asset, mode, isChecked) {
    console.log(`Trading toggle changed: ${asset} ${mode} ${isChecked}`);
    console.log('Dashboard object:', dashboard);
    console.log('Toggle element ID:', `${asset}-${mode}-toggle`);
    const toggleElement = document.getElementById(`${asset}-${mode}-toggle`);
    
    if (isChecked) {
        try {
            await dashboard.startAssetTrading(asset, mode);
        } catch (error) {
            console.error(`Failed to start ${asset} ${mode} trading:`, error);
            // Revert toggle on failure
            if (toggleElement) toggleElement.checked = false;
        }
    } else {
        try {
            await dashboard.stopAssetTrading(asset, mode);
        } catch (error) {
            console.error(`Failed to stop ${asset} ${mode} trading:`, error);
            // Revert toggle on failure
            if (toggleElement) toggleElement.checked = true;
        }
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
    window.handleTradingToggle = handleTradingToggle; // Make toggle handler globally accessible
    
    // Force an immediate connection check
    setTimeout(() => {
        if (dashboard) {
            console.log('Running initial connection check...');
            dashboard.checkBrokerConnections();
        }
    }, 1000);
    
    // Setup filters for new UI components
    dashboard.setupFilters();
    
    // Make sortPositions available globally for onclick handlers
    window.sortPositions = (asset, mode, column) => dashboard.sortPositions(asset, mode, column);
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

// Add missing updateWallet function
Dashboard.prototype.updateWallet = function(asset, mode, walletData) {
    if (!walletData) return;
    
    // Update balance display
    const balanceElement = document.getElementById(`${asset}-${mode}-balance`);
    if (balanceElement) {
        const balance = walletData.balance || 0;
        balanceElement.textContent = `$${balance.toFixed(2)}`;
        
        // Add color based on P&L
        if (walletData.pnl > 0) {
            balanceElement.style.color = 'var(--success)';
        } else if (walletData.pnl < 0) {
            balanceElement.style.color = 'var(--error)';
        } else {
            balanceElement.style.color = 'var(--text)';
        }
    }
    
    // Update P&L display if exists
    const pnlElement = document.getElementById(`${asset}-${mode}-pnl`);
    if (pnlElement && walletData.pnl !== undefined) {
        const pnl = walletData.pnl || 0;
        const pnlPercent = walletData.pnl_percent || 0;
        pnlElement.textContent = `${pnl >= 0 ? '+' : ''}$${pnl.toFixed(2)} (${pnlPercent.toFixed(2)}%)`;
        pnlElement.style.color = pnl >= 0 ? 'var(--success)' : 'var(--error)';
    }
};

// Add missing updateChart function for balance history
Dashboard.prototype.updateChart = function(asset, mode, walletData) {
    if (!walletData) return;
    
    const chartId = `${asset}-${mode}-chart`;
    const chartElement = document.getElementById(chartId);
    if (!chartElement) return;
    
    // Get or create chart instance
    let chart = this.charts[chartId];
    if (!chart) {
        const ctx = chartElement.getContext('2d');
        const isPositive = walletData.pnl >= 0;
        
        chart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Balance',
                    data: [],
                    borderColor: isPositive ? 'rgb(34, 197, 94)' : 'rgb(239, 68, 68)',
                    backgroundColor: isPositive ? 'rgba(34, 197, 94, 0.1)' : 'rgba(239, 68, 68, 0.1)',
                    borderWidth: 2,
                    tension: 0.1,
                    pointRadius: 0,
                    pointHoverRadius: 4,
                    fill: true
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                interaction: {
                    mode: 'index',
                    intersect: false
                },
                plugins: {
                    legend: {
                        display: false
                    },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                return 'Balance: $' + context.parsed.y.toFixed(2);
                            }
                        }
                    }
                },
                scales: {
                    x: {
                        display: true,
                        grid: {
                            display: false
                        },
                        ticks: {
                            color: '#9ca3af',
                            font: {
                                size: 10
                            },
                            maxRotation: 0,
                            autoSkip: true,
                            maxTicksLimit: 8
                        }
                    },
                    y: {
                        display: true,
                        beginAtZero: false,
                        grid: {
                            color: 'rgba(75, 85, 99, 0.2)'
                        },
                        ticks: {
                            color: '#9ca3af',
                            font: {
                                size: 10
                            },
                            callback: function(value) {
                                return '$' + value.toFixed(0);
                            }
                        }
                    }
                }
            }
        });
        this.charts[chartId] = chart;
    }
    
    // Update chart data
    const history = walletData.history || [];
    
    // If no history, create one from current balance
    if (history.length === 0 && walletData.balance !== undefined) {
        history.push({balance: walletData.balance, timestamp: new Date().toISOString()});
    }
    
    // Format labels based on time period
    chart.data.labels = history.map((h, i) => {
        if (h.timestamp) {
            const date = new Date(h.timestamp);
            const now = new Date();
            const diffMs = now - date;
            const diffHours = diffMs / (1000 * 60 * 60);
            
            if (diffHours < 24) {
                return date.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' });
            } else {
                return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
            }
        }
        return i === 0 ? 'Start' : `T+${i}`;
    });
    
    chart.data.datasets[0].data = history.map(h => h.balance || 0);
    
    // Update colors based on current P&L
    const isPositive = walletData.pnl >= 0;
    chart.data.datasets[0].borderColor = isPositive ? 'rgb(34, 197, 94)' : 'rgb(239, 68, 68)';
    chart.data.datasets[0].backgroundColor = isPositive ? 'rgba(34, 197, 94, 0.1)' : 'rgba(239, 68, 68, 0.1)';
    
    chart.update();
};

// Fix updateAssetControls to properly update toggle states
Dashboard.prototype.updateAssetControls = function(asset, status) {
    // Update paper trading toggle
    const paperToggle = document.getElementById(`${asset}-paper-toggle`);
    if (paperToggle) {
        paperToggle.checked = status.paper_trading_active || false;
    }
    
    // Update live trading toggle
    const liveToggle = document.getElementById(`${asset}-live-toggle`);
    if (liveToggle) {
        liveToggle.checked = status.live_trading_active || false;
    }
};

// Add training status display
Dashboard.prototype.updateTrainingStatus = function(asset, status) {
    // Update metrics if training is active
    if (status.paper_trading_active) {
        // Show training indicator
        const strategySection = document.querySelector(`#${asset}-paper-section .strategy-development`);
        if (strategySection) {
            const indicator = strategySection.querySelector('.training-indicator') || 
                            document.createElement('div');
            indicator.className = 'training-indicator';
            indicator.innerHTML = `
                <i class="fas fa-robot"></i> Training Active
                <span class="pulse"></span>
            `;
            if (!strategySection.querySelector('.training-indicator')) {
                strategySection.prepend(indicator);
            }
        }
        
        // Update trade count
        const tradesElement = document.getElementById(`${asset}-paper-trades`);
        if (tradesElement && status.paper_wallet) {
            const trades = status.paper_wallet.total_trades || 0;
            tradesElement.textContent = trades;
        }
    }
};