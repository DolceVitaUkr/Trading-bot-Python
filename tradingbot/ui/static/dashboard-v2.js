// Dashboard v2 JavaScript - Real Data Integration
class TradingDashboard {
    constructor() {
        this.charts = {};
        this.updateInterval = 3000; // 3 seconds
        this.assets = ['crypto', 'futures', 'forex', 'forex_options'];
        this.modes = ['paper', 'live'];
        this.priceUpdateInterval = 5000; // 5 seconds for price updates
        
        this.init();
    }

    init() {
        console.log('Initializing Trading Dashboard...');
        this.initCharts();
        this.startDataUpdates();
        this.bindEventListeners();
    }

    initCharts() {
        // Initialize charts for each asset and mode
        this.assets.forEach(asset => {
            this.modes.forEach(mode => {
                const canvasId = `${asset}-${mode}-chart`;
                const canvas = document.getElementById(canvasId);
                
                if (canvas) {
                    this.charts[`${asset}-${mode}`] = new Chart(canvas, {
                        type: 'line',
                        data: {
                            labels: [],
                            datasets: [{
                                label: 'Balance',
                                data: [],
                                borderColor: mode === 'paper' ? '#10b981' : '#3b82f6',
                                backgroundColor: mode === 'paper' ? 'rgba(16, 185, 129, 0.1)' : 'rgba(59, 130, 246, 0.1)',
                                tension: 0.4,
                                borderWidth: 2,
                                pointRadius: 0,
                                pointHoverRadius: 4,
                                fill: true
                            }]
                        },
                        options: {
                            responsive: true,
                            maintainAspectRatio: false,
                            plugins: {
                                legend: { display: false },
                                tooltip: {
                                    backgroundColor: 'rgba(0, 0, 0, 0.8)',
                                    titleColor: '#fff',
                                    bodyColor: '#fff',
                                    borderColor: mode === 'paper' ? '#10b981' : '#3b82f6',
                                    borderWidth: 1,
                                    displayColors: false,
                                    callbacks: {
                                        label: function(context) {
                                            return '$' + context.raw.toFixed(2);
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
                                    grid: { display: false }
                                }
                            },
                            interaction: {
                                intersect: false,
                                mode: 'index'
                            }
                        }
                    });
                }
            });
        });
    }

    async startDataUpdates() {
        // Initial load
        await this.updateAllData();
        
        // Set up periodic updates
        setInterval(() => this.updateAllData(), this.updateInterval);
        
        // Set up price updates (less frequent)
        setInterval(() => this.updatePrices(), this.priceUpdateInterval);
    }

    async updateAllData() {
        for (const asset of this.assets) {
            try {
                await this.updateAssetData(asset);
            } catch (error) {
                console.error(`Error updating ${asset} data:`, error);
            }
        }
        
        // Update global stats
        await this.updateGlobalStats();
    }

    async updateAssetData(asset) {
        try {
            const response = await fetch(`/asset/${asset}/status`);
            const data = await response.json();
            
            // Update Paper Trading Data
            if (data.paper_wallet) {
                this.updateModeData(asset, 'paper', data.paper_wallet);
                this.updateChart(asset, 'paper', data.paper_wallet.history);
            }
            
            // Update Live Trading Data
            if (data.live_wallet) {
                this.updateModeData(asset, 'live', data.live_wallet);
                this.updateChart(asset, 'live', data.live_wallet.history);
            }
            
            // Update positions
            await this.updatePositions(asset);
            
            // Update strategies
            await this.updateStrategies(asset);
            
        } catch (error) {
            console.error(`Failed to update ${asset} data:`, error);
        }
    }

    updateModeData(asset, mode, walletData) {
        const prefix = `${asset}-${mode}`;
        
        // Update balance displays
        const totalEl = document.querySelector(`#${prefix}-section .balance-item:nth-child(1) .balance-value`);
        const availableEl = document.querySelector(`#${prefix}-section .balance-item:nth-child(2) .balance-value`);
        const usedEl = document.querySelector(`#${prefix}-section .balance-item:nth-child(3) .balance-value`);
        const rewardEl = document.querySelector(`#${prefix}-section .balance-item:nth-child(4) .balance-value`);
        
        if (totalEl) totalEl.textContent = `$${walletData.balance.toFixed(2)}`;
        if (availableEl) availableEl.textContent = `$${(walletData.available_balance || walletData.balance).toFixed(2)}`;
        if (usedEl) usedEl.textContent = `$${(walletData.used_in_positions || 0).toFixed(2)}`;
        if (rewardEl) rewardEl.textContent = walletData.reward_points || '0';
        
        // Update P&L if available
        if (walletData.pnl !== undefined) {
            const pnlValue = walletData.pnl;
            const pnlPercent = walletData.pnl_percent || 0;
            
            // You can add P&L display elements to the HTML and update them here
        }
    }

    updateChart(asset, mode, history) {
        const chartKey = `${asset}-${mode}`;
        const chart = this.charts[chartKey];
        
        if (!chart || !history || history.length === 0) return;
        
        // Extract labels and data from history
        const labels = history.map(point => {
            if (point.timestamp) {
                const date = new Date(point.timestamp);
                return date.toLocaleTimeString();
            }
            return '';
        });
        
        const data = history.map(point => point.balance || 0);
        
        // Update chart data
        chart.data.labels = labels;
        chart.data.datasets[0].data = data;
        chart.update('none'); // Update without animation for performance
    }

    async updatePositions(asset) {
        try {
            const response = await fetch(`/asset/${asset}/positions`);
            const data = await response.json();
            
            // Update paper positions
            if (data.paper_positions) {
                this.updatePositionsTable(asset, 'paper', data.paper_positions);
            }
            
            // Update live positions
            if (data.live_positions) {
                this.updatePositionsTable(asset, 'live', data.live_positions);
            }
        } catch (error) {
            console.error(`Failed to update ${asset} positions:`, error);
        }
    }

    updatePositionsTable(asset, mode, positions) {
        const tableId = `${asset}-${mode}-positions`;
        const tbody = document.getElementById(tableId);
        
        if (!tbody) return;
        
        if (positions.length === 0) {
            tbody.innerHTML = '<tr><td colspan="6" style="text-align: center; color: #6c757d;">No active positions</td></tr>';
            return;
        }
        
        tbody.innerHTML = positions.map(pos => `
            <tr>
                <td>${new Date(pos.entry_time).toLocaleString()}</td>
                <td>${pos.symbol}</td>
                <td class="${pos.side === 'buy' ? 'text-success' : 'text-danger'}">${pos.side.toUpperCase()}</td>
                <td>${pos.size.toFixed(4)}</td>
                <td>$${(pos.entry_price * pos.size).toFixed(2)}</td>
                <td class="${pos.pnl >= 0 ? 'text-success' : 'text-danger'}">
                    $${pos.pnl.toFixed(2)} (${pos.pnl_percent.toFixed(2)}%)
                </td>
            </tr>
        `).join('');
    }

    async updateStrategies(asset) {
        try {
            const response = await fetch(`/asset/${asset}/strategies`);
            const data = await response.json();
            
            // Update strategy displays if needed
            // This could show active strategies, their performance, etc.
        } catch (error) {
            console.error(`Failed to update ${asset} strategies:`, error);
        }
    }

    async updateGlobalStats() {
        try {
            const response = await fetch('/stats/global');
            const data = await response.json();
            
            // Update header stats
            document.getElementById('live-wallet').textContent = `$${data.total_live_balance.toFixed(2)}`;
            document.getElementById('live-pnl').textContent = `${data.live_pnl >= 0 ? '+' : ''}$${data.live_pnl.toFixed(2)}`;
            document.getElementById('live-session').textContent = `$${data.live_session_pnl.toFixed(2)}`;
            
            document.getElementById('paper-wallet').textContent = `$${data.total_paper_balance.toFixed(2)}`;
            document.getElementById('paper-pnl').textContent = `${data.paper_pnl >= 0 ? '+' : ''}$${data.paper_pnl.toFixed(2)}`;
            document.getElementById('paper-session').textContent = `$${data.paper_session_pnl.toFixed(2)}`;
            
            // Update P&L colors
            document.getElementById('live-pnl').className = `stat-value ${data.live_pnl >= 0 ? 'positive' : 'negative'}`;
            document.getElementById('paper-pnl').className = `stat-value ${data.paper_pnl >= 0 ? 'positive' : 'negative'}`;
            
        } catch (error) {
            console.error('Failed to update global stats:', error);
        }
    }

    async updatePrices() {
        // Update live price tickers
        const priceMappings = [
            { asset: 'crypto', symbol: 'BTCUSDT', elementId: 'btc-price', changeId: 'btc-change' },
            { asset: 'futures', symbol: 'BTCUSDT', elementId: 'futures-price', changeId: 'futures-change' },
            { asset: 'forex', symbol: 'EURUSD', elementId: 'forex-price', changeId: 'forex-change' },
            { asset: 'options', symbol: 'SPX', elementId: 'options-price', changeId: 'options-change' }
        ];
        
        for (const config of priceMappings) {
            try {
                const response = await fetch(`/price/${config.symbol}`);
                const data = await response.json();
                
                // Update price display
                const priceEl = document.getElementById(config.elementId);
                const changeEl = document.getElementById(config.changeId);
                
                if (priceEl) {
                    const formattedPrice = config.asset === 'forex' ? 
                        data.price.toFixed(4) : 
                        `$${data.price.toFixed(2)}`;
                    priceEl.textContent = formattedPrice;
                }
                
                if (changeEl) {
                    const change = data.change_24h || 0;
                    changeEl.textContent = `${change >= 0 ? '+' : ''}${change.toFixed(2)}%`;
                    changeEl.className = `ticker-change ${change >= 0 ? 'positive' : 'negative'}`;
                }
            } catch (error) {
                console.error(`Failed to update ${config.asset} price:`, error);
            }
        }
    }

    bindEventListeners() {
        // Bind toggle switches
        this.assets.forEach(asset => {
            this.modes.forEach(mode => {
                const toggleId = `${asset}-${mode}-toggle`;
                const toggle = document.getElementById(toggleId);
                
                if (toggle) {
                    toggle.addEventListener('change', (e) => {
                        this.handleToggle(asset, mode, e.target.checked);
                    });
                }
            });
        });
        
        // Emergency stop button
        const emergencyBtn = document.querySelector('.emergency-stop');
        if (emergencyBtn) {
            emergencyBtn.addEventListener('click', () => this.handleEmergencyStop());
        }
        
        // Tab switching
        document.querySelectorAll('.nav-tab').forEach(tab => {
            tab.addEventListener('click', (e) => this.switchTab(e));
        });
    }

    async handleToggle(asset, mode, enabled) {
        try {
            const endpoint = mode === 'paper' ? `/paper/${asset}/${enabled ? 'enable' : 'disable'}` : `/live/${asset}/${enabled ? 'enable' : 'disable'}`;
            
            const response = await fetch(endpoint, { method: 'POST' });
            const data = await response.json();
            
            if (!response.ok) {
                throw new Error(data.detail || 'Failed to toggle trading');
            }
            
            console.log(`${asset} ${mode} trading ${enabled ? 'enabled' : 'disabled'}`);
            
            // Update UI to reflect the change
            this.updateAssetData(asset);
            
        } catch (error) {
            console.error('Failed to toggle trading:', error);
            // Revert the toggle
            document.getElementById(`${asset}-${mode}-toggle`).checked = !enabled;
            alert(`Failed to ${enabled ? 'enable' : 'disable'} ${mode} trading for ${asset}: ${error.message}`);
        }
    }

    async handleEmergencyStop() {
        if (!confirm('Are you sure you want to emergency stop all trading activities?')) {
            return;
        }
        
        try {
            const response = await fetch('/emergency-stop', { method: 'POST' });
            const data = await response.json();
            
            console.log('Emergency stop activated:', data);
            alert('Emergency stop activated! All trading has been halted.');
            
            // Refresh the page to show updated state
            setTimeout(() => window.location.reload(), 1000);
            
        } catch (error) {
            console.error('Failed to activate emergency stop:', error);
            alert('Failed to activate emergency stop! Please try again.');
        }
    }

    switchTab(event) {
        const tabElement = event.target.closest('.nav-tab');
        const tabName = tabElement.textContent.trim().toLowerCase();
        
        // Update nav tabs
        document.querySelectorAll('.nav-tab').forEach(tab => {
            tab.classList.remove('active');
        });
        tabElement.classList.add('active');
        
        // Update content
        document.querySelectorAll('.tab-content').forEach(content => {
            content.classList.remove('active');
        });
        
        const contentId = `${tabName}-tab`;
        const contentElement = document.getElementById(contentId);
        if (contentElement) {
            contentElement.classList.add('active');
        }
    }
}

// Initialize dashboard when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.dashboard = new TradingDashboard();
});

// Activity log function
function addActivity(message, type = 'info') {
    const activityLog = document.getElementById('activity-log');
    if (!activityLog) return;
    
    const activityItem = document.createElement('div');
    activityItem.className = 'activity-item';
    activityItem.innerHTML = `
        <div>${message}</div>
        <div class="activity-time">${new Date().toLocaleTimeString()}</div>
    `;
    
    // Remove "no activity" message if present
    const noActivity = activityLog.querySelector('.activity-item');
    if (noActivity && noActivity.textContent.includes('System initialized')) {
        noActivity.remove();
    }
    
    activityLog.insertBefore(activityItem, activityLog.firstChild);
    
    // Keep only last 50 activities
    while (activityLog.children.length > 50) {
        activityLog.removeChild(activityLog.lastChild);
    }
}