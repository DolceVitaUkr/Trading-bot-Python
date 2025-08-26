class TradingBotDashboard {
    constructor() {
        this.baseUrl = '';
        this.updateInterval = 5000; // 5 seconds
        this.updateTimer = null;
        this.currentSection = 'dashboard';
        this.init();
    }

    init() {
        this.bindEvents();
        this.loadInitialData();
        this.startPeriodicUpdates();
    }

    bindEvents() {
        // Navigation
        document.querySelectorAll('.nav-item').forEach(item => {
            item.addEventListener('click', (e) => {
                e.preventDefault();
                const section = item.dataset.section;
                if (section) {
                    console.log(`Nav item clicked: ${section}`);
                    this.switchSection(section);
                } else {
                    console.error('Nav item missing data-section attribute:', item);
                }
            });
        });

        // Kill switch
        const killSwitch = document.getElementById('globalKillSwitch');
        if (killSwitch) {
            killSwitch.addEventListener('change', this.handleKillSwitch.bind(this));
        }

        // Settings kill switch
        const settingsKillSwitch = document.getElementById('settingsKillSwitch');
        if (settingsKillSwitch) {
            settingsKillSwitch.addEventListener('change', this.handleKillSwitch.bind(this));
        }

        // Trading mode toggle
        const tradingModeToggle = document.getElementById('tradingModeToggle');
        if (tradingModeToggle) {
            tradingModeToggle.addEventListener('change', this.handleTradingModeToggle.bind(this));
        }

        // Trading pair toggles
        document.addEventListener('click', (e) => {
            if (e.target.classList.contains('toggle-btn')) {
                this.handlePairToggle(e.target);
            }
        });

        // Refresh pairs button
        const refreshPairsBtn = document.getElementById('refreshPairsBtn');
        if (refreshPairsBtn) {
            refreshPairsBtn.addEventListener('click', () => this.loadTopPairs());
        }

        // Telegram settings
        const saveTelegramBtn = document.getElementById('saveTelegramBtn');
        if (saveTelegramBtn) {
            saveTelegramBtn.addEventListener('click', () => this.saveTelegramSettings());
        }

        // Add pair modal (keeping for manual addition)
        const addPairBtn = document.getElementById('addPairBtn');
        const modal = document.getElementById('addPairModal');
        const modalClose = document.querySelector('.modal-close');
        const cancelBtn = document.getElementById('cancelAddPair');
        const confirmBtn = document.getElementById('confirmAddPair');

        if (addPairBtn) {
            addPairBtn.addEventListener('click', () => this.openAddPairModal());
        }

        if (modalClose) {
            modalClose.addEventListener('click', () => this.closeAddPairModal());
        }

        if (cancelBtn) {
            cancelBtn.addEventListener('click', () => this.closeAddPairModal());
        }

        if (confirmBtn) {
            confirmBtn.addEventListener('click', () => this.addNewPair());
        }

        // Close modal on backdrop click
        if (modal) {
            modal.addEventListener('click', (e) => {
                if (e.target === modal) {
                    this.closeAddPairModal();
                }
            });
        }

        // Enter key on pair input
        const pairInput = document.getElementById('newPairInput');
        if (pairInput) {
            pairInput.addEventListener('keypress', (e) => {
                if (e.key === 'Enter') {
                    this.addNewPair();
                }
            });
        }
    }

    switchSection(section) {
        try {
            console.log(`Switching to section: ${section}`);
            
            // Update navigation
            document.querySelectorAll('.nav-item').forEach(item => {
                item.classList.remove('active');
            });
            const navItem = document.querySelector(`[data-section="${section}"]`);
            if (navItem) {
                navItem.classList.add('active');
            }

            // Hide all sections
            document.querySelectorAll('.section-content').forEach(content => {
                content.classList.add('hidden');
            });

            // Show selected section
            const sectionElement = document.getElementById(`${section}-section`);
            if (sectionElement) {
                sectionElement.classList.remove('hidden');
            } else {
                console.error(`Section element not found: ${section}-section`);
                return;
            }

            // Update header
            const titles = {
                dashboard: { title: 'Dashboard', description: "Welcome back! Here's what's happening with your bot today." },
                trading: { title: 'Trading', description: 'Configure trading modes and monitor active positions.' },
                portfolio: { title: 'Portfolio', description: 'View your portfolio performance and asset allocation.' },
                settings: { title: 'Settings', description: 'Configure bot settings, notifications, and safety parameters.' }
            };

            const sectionInfo = titles[section];
            if (sectionInfo) {
                const titleEl = document.getElementById('pageTitle');
                const descEl = document.getElementById('pageDescription');
                if (titleEl) titleEl.textContent = sectionInfo.title;
                if (descEl) descEl.textContent = sectionInfo.description;
            }

            this.currentSection = section;

            // Load section-specific data
            if (section === 'dashboard') {
                this.loadTopPairs();
                this.loadRecentActivity();
            } else if (section === 'settings') {
                this.loadTelegramConfig();
            } else if (section === 'trading') {
                this.loadTradingMode();
                this.loadTradingStatus();
            } else if (section === 'portfolio') {
                this.loadPortfolioPositions();
            }
            
            // Add activity log for successful navigation
            this.addActivity(`Switched to ${section} section`, 'info');
            
        } catch (error) {
            console.error('Error switching sections:', error);
            this.addActivity(`Failed to switch to ${section} section`, 'error');
        }
    }

    async loadInitialData() {
        try {
            await this.updateBotStatus();
            await this.updateStats();
            await this.loadTopPairs();
            await this.loadRecentActivity();
            await this.loadTradingMode();
            await this.loadTelegramConfig();
            this.addActivity('Bot Dashboard Loaded', 'info');
        } catch (error) {
            console.error('Failed to load initial data:', error);
            this.addActivity('Failed to connect to bot API', 'error');
        }
    }

    startPeriodicUpdates() {
        this.updateTimer = setInterval(() => {
            this.updateBotStatus();
            this.updateStats();
            if (this.currentSection === 'dashboard') {
                this.loadRecentActivity();
            }
        }, this.updateInterval);
    }

    stopPeriodicUpdates() {
        if (this.updateTimer) {
            clearInterval(this.updateTimer);
            this.updateTimer = null;
        }
    }

    async updateBotStatus() {
        try {
            // Get both general status and detailed bot status
            const [statusResponse, botStatusResponse] = await Promise.all([
                fetch(`${this.baseUrl}/status`),
                fetch(`${this.baseUrl}/bot/status`)
            ]);
            
            const statusData = await statusResponse.json();
            const botStatusData = await botStatusResponse.json();
            
            this.updateStatusIndicator(botStatusData.status === 'online');
            this.updateKillSwitchStatus(statusData.global?.kill_switch || false);
            this.updateLastUpdate();
            this.updateCurrentActivity(botStatusData.current_activity || 'Unknown');
            
            return { status: statusData, botStatus: botStatusData };
        } catch (error) {
            console.error('Failed to fetch bot status:', error);
            this.updateStatusIndicator(false);
            this.updateCurrentActivity('Service unavailable');
            throw error;
        }
    }

    async updateStats() {
        try {
            const response = await fetch(`${this.baseUrl}/stats`);
            const data = await response.json();
            
            // Update total P&L
            const totalPnl = data.total_pnl || 0;
            document.getElementById('totalPnl').textContent = `$${totalPnl.toLocaleString('en-US', {minimumFractionDigits: 2, maximumFractionDigits: 2})}`;
            
            const pnlChange = data.pnl_change || 0;
            const pnlChangeEl = document.getElementById('pnlChange');
            pnlChangeEl.textContent = `${pnlChange >= 0 ? '+' : ''}${pnlChange.toFixed(2)}%`;
            pnlChangeEl.className = `stat-change ${pnlChange >= 0 ? 'positive' : 'negative'}`;
            
            // Update active trades
            document.getElementById('activeTrades').textContent = data.active_trades || 0;
            document.getElementById('tradesChange').textContent = `${data.trades_today || 0} today`;
            
            // Update win rate
            const winRate = data.win_rate || 0;
            document.getElementById('winRate').textContent = `${(winRate * 100).toFixed(0)}%`;
            document.getElementById('winRateChange').textContent = 'Last 24h';
            
            // Update balance
            const balance = data.balance || 0;
            document.getElementById('balance').textContent = `$${balance.toLocaleString('en-US', {minimumFractionDigits: 2, maximumFractionDigits: 2})}`;
            document.getElementById('balanceChange').textContent = 'Available';
            
        } catch (error) {
            console.error('Failed to update stats:', error);
            // Fall back to showing zeros
            document.getElementById('totalPnl').textContent = '$0.00';
            document.getElementById('pnlChange').textContent = '+0.00%';
            document.getElementById('activeTrades').textContent = '0';
            document.getElementById('tradesChange').textContent = '0 today';
            document.getElementById('winRate').textContent = '0%';
            document.getElementById('balance').textContent = '$0.00';
        }
    }

    async loadTopPairs() {
        try {
            const response = await fetch(`${this.baseUrl}/pairs/top`);
            const data = await response.json();
            
            const pairsList = document.getElementById('pairsList');
            pairsList.innerHTML = '';
            
            data.pairs.forEach(pair => {
                const pairItem = document.createElement('div');
                pairItem.className = 'pair-item enhanced';
                
                const changeClass = pair.change24h >= 0 ? 'positive' : 'negative';
                const changeSign = pair.change24h >= 0 ? '+' : '';
                
                // Score color based on value
                let scoreClass = 'score-low';
                if (pair.score > 80) scoreClass = 'score-high';
                else if (pair.score > 70) scoreClass = 'score-medium';
                
                // Regime badge color
                const regimeColors = {
                    'trending': 'regime-trending',
                    'breakout': 'regime-breakout', 
                    'volatile': 'regime-volatile',
                    'ranging': 'regime-ranging'
                };
                const regimeClass = regimeColors[pair.regime] || 'regime-ranging';
                
                pairItem.innerHTML = `
                    <div class="pair-info">
                        <div class="pair-header">
                            <div class="pair-symbol">${pair.symbol}</div>
                            <div class="pair-score ${scoreClass}" title="Multi-factor score: ${pair.score}/100">
                                ${pair.score || 0}
                            </div>
                        </div>
                        <div class="pair-details">
                            <div class="pair-price">$${pair.price.toLocaleString()}</div>
                            <div class="pair-regime ${regimeClass}">${pair.regime || 'ranging'}</div>
                        </div>
                        <div class="pair-metrics">
                            <div class="pair-volume">Vol: $${(pair.volume24h / 1000000).toFixed(1)}M</div>
                            <div class="pair-change ${changeClass}">${changeSign}${pair.change24h.toFixed(2)}%</div>
                        </div>
                        ${pair.atr_pct ? `
                        <div class="pair-advanced" title="Advanced metrics">
                            <span>ATR: ${pair.atr_pct}%</span>
                            <span>Spread: ${pair.spread_bps}bp</span>
                        </div>` : ''}
                    </div>
                    <div class="pair-status">
                        <span class="status-badge disabled">Disabled</span>
                        <button class="toggle-btn" data-pair="${pair.symbol}">Enable</button>
                    </div>
                `;
                
                // Add click handler for score breakdown
                if (pair.score) {
                    const scoreElement = pairItem.querySelector('.pair-score');
                    scoreElement.style.cursor = 'pointer';
                    scoreElement.addEventListener('click', () => {
                        this.showScoreBreakdown(pair);
                    });
                }
                
                pairsList.appendChild(pairItem);
            });
        } catch (error) {
            console.error('Failed to load top pairs:', error);
            document.getElementById('pairsList').innerHTML = '<div class="loading-message">Failed to load pairs</div>';
        }
    }

    async loadRecentActivity() {
        try {
            const response = await fetch(`${this.baseUrl}/activity/recent`);
            const data = await response.json();
            
            const activityList = document.getElementById('activityList');
            activityList.innerHTML = '';
            
            data.activities.forEach(activity => {
                this.addActivityToList(activity.message, activity.type, activity.timestamp);
            });
        } catch (error) {
            console.error('Failed to load recent activity:', error);
            document.getElementById('activityList').innerHTML = '<div class="loading-message">Failed to load activity</div>';
        }
    }

    async loadTradingMode() {
        try {
            const response = await fetch(`${this.baseUrl}/config/trading-mode`);
            const data = await response.json();
            
            const toggle = document.getElementById('tradingModeToggle');
            const label = document.getElementById('tradingModeLabel');
            const alert = document.getElementById('tradingModeAlert');
            const balance = document.getElementById('paperBalance');
            const statusEl = document.getElementById('tradingModeStatus');
            
            if (toggle) {
                toggle.checked = data.mode === 'live';
            }
            
            if (label) {
                label.textContent = data.mode === 'live' ? 'Live Trading' : 'Paper Trading';
            }
            
            if (alert) {
                if (data.mode === 'live') {
                    alert.className = 'alert warning';
                    alert.innerHTML = `
                        <i class="fas fa-exclamation-triangle"></i>
                        <span>LIVE TRADING MODE - Real money is at risk!</span>
                    `;
                } else {
                    alert.className = 'alert';
                    alert.innerHTML = `
                        <i class="fas fa-info-circle"></i>
                        <span>Currently in Paper Trading mode. No real money is at risk.</span>
                    `;
                }
            }
            
            if (balance) {
                balance.textContent = `$${data.paper_balance.toLocaleString()}`;
            }
            
            if (statusEl) {
                statusEl.textContent = data.mode === 'live' ? 'Live' : 'Paper';
                statusEl.style.color = data.mode === 'live' ? 'var(--warning-color)' : 'var(--text-primary)';
            }
        } catch (error) {
            console.error('Failed to load trading mode:', error);
        }
    }

    async loadTelegramConfig() {
        try {
            const response = await fetch(`${this.baseUrl}/config/telegram`);
            const data = await response.json();
            
            const status = document.getElementById('telegramConfigStatus');
            const systemStatus = document.getElementById('telegramStatus');
            const botTokenInput = document.getElementById('telegramBotToken');
            const chatIdInput = document.getElementById('telegramChatId');
            
            if (status) {
                if (data.enabled) {
                    status.className = 'telegram-status enabled';
                    status.innerHTML = `
                        <i class="fas fa-check-circle"></i>
                        <span>Configured and enabled (using config.json values)</span>
                    `;
                } else {
                    status.className = 'telegram-status disabled';
                    status.innerHTML = `
                        <i class="fas fa-times-circle"></i>
                        <span>Not configured in config.json</span>
                    `;
                }
            }
            
            if (systemStatus) {
                systemStatus.textContent = data.enabled ? 'Enabled' : 'Disabled';
                systemStatus.style.color = data.enabled ? 'var(--success-color)' : 'var(--danger-color)';
            }
            
            // Show preview of existing config
            if (botTokenInput && data.token_preview) {
                botTokenInput.placeholder = `Current: ${data.token_preview}`;
            }
            if (chatIdInput && data.chat_id_value) {
                chatIdInput.placeholder = `Current: ${data.chat_id_value}`;
            }
        } catch (error) {
            console.error('Failed to load telegram config:', error);
        }
    }

    updateStatusIndicator(isOnline) {
        const indicator = document.getElementById('botStatus');
        const dot = indicator.querySelector('.status-dot');
        const text = indicator.querySelector('span');
        
        if (isOnline) {
            indicator.classList.add('online');
            text.textContent = 'Online';
        } else {
            indicator.classList.remove('online');
            text.textContent = 'Offline';
        }
    }

    updateKillSwitchStatus(isEnabled) {
        const killSwitch = document.getElementById('globalKillSwitch');
        const settingsKillSwitch = document.getElementById('settingsKillSwitch');
        const statusText = document.getElementById('killSwitchStatus');
        
        if (killSwitch) {
            killSwitch.checked = isEnabled;
        }
        
        if (settingsKillSwitch) {
            settingsKillSwitch.checked = isEnabled;
        }
        
        if (statusText) {
            statusText.textContent = isEnabled ? 'On' : 'Off';
            statusText.style.color = isEnabled ? 'var(--danger-color)' : 'var(--text-primary)';
        }
    }

    updateLastUpdate() {
        const lastUpdateEl = document.getElementById('lastUpdate');
        if (lastUpdateEl) {
            const now = new Date();
            const timeString = now.toLocaleTimeString();
            lastUpdateEl.textContent = timeString;
        }
    }

    updateCurrentActivity(activity) {
        // Update bot activity status display
        const activityEl = document.getElementById('currentActivity');
        if (activityEl) {
            activityEl.textContent = activity;
        }
        
        // Also show in system status if available
        const statusElements = document.querySelectorAll('.bot-activity');
        statusElements.forEach(el => {
            el.textContent = activity;
        });
    }

    async loadTradingStatus() {
        try {
            const response = await fetch(`${this.baseUrl}/trading/status`);
            const data = await response.json();
            
            // Update trading controls in the UI
            this.updateTradingControls(data.trading_status || {});
        } catch (error) {
            console.error('Failed to load trading status:', error);
        }
    }

    async loadPortfolioPositions() {
        try {
            const response = await fetch(`${this.baseUrl}/portfolio/positions`);
            const data = await response.json();
            
            // Update portfolio display
            this.updatePortfolioDisplay(data);
        } catch (error) {
            console.error('Failed to load portfolio positions:', error);
        }
    }

    updateTradingControls(tradingStatus) {
        // This will be used to update the trading section controls
        console.log('Trading status:', tradingStatus);
    }

    updatePortfolioDisplay(portfolioData) {
        // Update paper portfolio balance
        const paperBalance = document.getElementById('paperPortfolioBalance');
        if (paperBalance && portfolioData.paper_balance !== undefined) {
            paperBalance.textContent = `$${portfolioData.paper_balance.toLocaleString('en-US', {minimumFractionDigits: 2, maximumFractionDigits: 2})}`;
        }
        
        // Update live portfolio balance
        const liveBalance = document.getElementById('livePortfolioBalance');
        if (liveBalance && portfolioData.live_balance !== undefined) {
            liveBalance.textContent = `$${portfolioData.live_balance.toLocaleString('en-US', {minimumFractionDigits: 2, maximumFractionDigits: 2})}`;
        }
        
        // Update positions (would populate with actual position data)
        console.log('Portfolio data:', portfolioData);
    }

    async startTrading(assetType) {
        try {
            const response = await fetch(`${this.baseUrl}/trading/start/${assetType}`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                }
            });
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            const data = await response.json();
            this.updateAssetStatus(assetType, 'running');
            this.addActivity(`${assetType.toUpperCase()} trading started in ${data.mode} mode`, 'success');
            
        } catch (error) {
            console.error(`Failed to start ${assetType} trading:`, error);
            this.addActivity(`Failed to start ${assetType} trading`, 'error');
        }
    }

    async stopTrading(assetType) {
        try {
            const response = await fetch(`${this.baseUrl}/trading/stop/${assetType}`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                }
            });
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            const data = await response.json();
            this.updateAssetStatus(assetType, 'stopped');
            this.addActivity(`${assetType.toUpperCase()} trading stopped`, 'info');
            
        } catch (error) {
            console.error(`Failed to stop ${assetType} trading:`, error);
            this.addActivity(`Failed to stop ${assetType} trading`, 'error');
        }
    }

    updateAssetStatus(assetType, status) {
        const statusElement = document.getElementById(`${assetType}-status`);
        const startButton = document.getElementById(`${assetType}-start`);
        const stopButton = document.getElementById(`${assetType}-stop`);
        
        if (statusElement) {
            if (status === 'running') {
                statusElement.textContent = 'Running';
                statusElement.className = 'status-badge enabled';
            } else {
                statusElement.textContent = 'Stopped';
                statusElement.className = 'status-badge disabled';
            }
        }
        
        if (startButton && stopButton) {
            if (status === 'running') {
                startButton.classList.add('hidden');
                stopButton.classList.remove('hidden');
            } else {
                startButton.classList.remove('hidden');
                stopButton.classList.add('hidden');
            }
        }
    }

    async handleKillSwitch(event) {
        const isEnabled = event.target.checked;
        
        try {
            const response = await fetch(`${this.baseUrl}/kill/global/${isEnabled ? 'on' : 'off'}`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                }
            });
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            const data = await response.json();
            this.updateKillSwitchStatus(data.kill_switch);
            
            const action = isEnabled ? 'enabled' : 'disabled';
            this.addActivity(`Kill switch ${action}`, isEnabled ? 'warning' : 'info');
            
        } catch (error) {
            console.error('Failed to toggle kill switch:', error);
            // Revert the switch
            event.target.checked = !isEnabled;
            this.addActivity('Failed to toggle kill switch', 'error');
        }
    }

    async handleTradingModeToggle(event) {
        const isLive = event.target.checked;
        const mode = isLive ? 'live' : 'paper';
        
        try {
            const response = await fetch(`${this.baseUrl}/config/trading-mode`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ mode })
            });
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            await this.loadTradingMode();
            this.addActivity(`Switched to ${mode} trading mode`, 'info');
            
        } catch (error) {
            console.error('Failed to toggle trading mode:', error);
            // Revert the toggle
            event.target.checked = !isLive;
            this.addActivity('Failed to change trading mode', 'error');
        }
    }

    async handlePairToggle(button) {
        const pair = button.dataset.pair;
        const isCurrentlyEnabled = button.textContent.trim() === 'Disable';
        const newAction = isCurrentlyEnabled ? 'disable' : 'enable';
        
        // Disable button during request
        button.disabled = true;
        button.textContent = 'Loading...';
        
        try {
            const response = await fetch(`${this.baseUrl}/live/${pair}/${newAction}`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                }
            });
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            const data = await response.json();
            
            // Update UI
            this.updatePairStatus(pair, data.live);
            
            const action = data.live ? 'enabled' : 'disabled';
            this.addActivity(`${pair} trading ${action}`, 'info');
            
        } catch (error) {
            console.error(`Failed to ${newAction} ${pair}:`, error);
            this.addActivity(`Failed to ${newAction} ${pair}`, 'error');
        } finally {
            button.disabled = false;
        }
    }

    async saveTelegramSettings() {
        // Show message that telegram is configured via config.json
        alert('Telegram is configured via config.json file. Bot token and chat ID are already set from your configuration.');
        
        // Clear the form
        document.getElementById('telegramBotToken').value = '';
        document.getElementById('telegramChatId').value = '';
        
        await this.loadTelegramConfig();
        this.addActivity('Telegram configuration loaded from config.json', 'info');
    }

    updatePairStatus(pair, isEnabled) {
        const pairItems = document.querySelectorAll('.pair-item');
        
        pairItems.forEach(item => {
            const symbol = item.querySelector('.pair-symbol').textContent;
            if (symbol === pair) {
                const badge = item.querySelector('.status-badge');
                const button = item.querySelector('.toggle-btn');
                
                if (isEnabled) {
                    badge.textContent = 'Enabled';
                    badge.className = 'status-badge enabled';
                    button.textContent = 'Disable';
                } else {
                    badge.textContent = 'Disabled';
                    badge.className = 'status-badge disabled';
                    button.textContent = 'Enable';
                }
            }
        });
    }

    openAddPairModal() {
        const modal = document.getElementById('addPairModal');
        const input = document.getElementById('newPairInput');
        
        if (modal) {
            modal.style.display = 'block';
            if (input) {
                input.value = '';
                input.focus();
            }
        }
    }

    closeAddPairModal() {
        const modal = document.getElementById('addPairModal');
        if (modal) {
            modal.style.display = 'none';
        }
    }

    async addNewPair() {
        const input = document.getElementById('newPairInput');
        const pair = input.value.trim().toUpperCase();
        
        if (!pair) {
            alert('Please enter a trading pair');
            return;
        }
        
        if (!this.validatePairFormat(pair)) {
            alert('Invalid pair format. Please use format like BTC/USDT');
            return;
        }
        
        try {
            // Add the pair to the UI
            this.addPairToList(pair);
            this.closeAddPairModal();
            this.addActivity(`Added new trading pair: ${pair}`, 'info');
            
        } catch (error) {
            console.error('Failed to add pair:', error);
            this.addActivity(`Failed to add pair: ${pair}`, 'error');
        }
    }

    validatePairFormat(pair) {
        // Basic validation for trading pairs
        return /^[A-Z]{2,10}\/[A-Z]{2,10}$/.test(pair);
    }

    addPairToList(pair) {
        const pairsList = document.getElementById('pairsList');
        const pairItem = document.createElement('div');
        pairItem.className = 'pair-item';
        pairItem.innerHTML = `
            <div class="pair-info">
                <div class="pair-symbol">${pair}</div>
                <div class="pair-price">$0.00</div>
                <div class="pair-volume">Vol: $0M</div>
                <div class="pair-change">0.00%</div>
            </div>
            <div class="pair-status">
                <span class="status-badge disabled">Disabled</span>
                <button class="toggle-btn" data-pair="${pair}">Enable</button>
            </div>
        `;
        
        pairsList.appendChild(pairItem);
    }

    addActivity(message, type = 'info') {
        this.addActivityToList(message, type, new Date().toISOString());
    }

    addActivityToList(message, type = 'info', timestamp = null) {
        const activityList = document.getElementById('activityList');
        const activityItem = document.createElement('div');
        activityItem.className = 'activity-item';
        
        let iconClass = 'fas fa-info-circle';
        let iconType = '';
        
        if (type === 'warning') {
            iconClass = 'fas fa-exclamation-triangle';
            iconType = 'warning';
        } else if (type === 'error') {
            iconClass = 'fas fa-exclamation-circle';
            iconType = 'error';
        } else if (type === 'success') {
            iconClass = 'fas fa-check-circle';
            iconType = 'success';
        }
        
        const timeString = timestamp ? 
            new Date(timestamp).toLocaleTimeString() : 
            'Just now';
        
        activityItem.innerHTML = `
            <div class="activity-icon ${iconType}">
                <i class="${iconClass}"></i>
            </div>
            <div class="activity-content">
                <div class="activity-title">${message}</div>
                <div class="activity-time">${timeString}</div>
            </div>
        `;
        
        // Add to top of list
        activityList.insertBefore(activityItem, activityList.firstChild);
        
        // Keep only last 10 activities
        while (activityList.children.length > 10) {
            activityList.removeChild(activityList.lastChild);
        }
    }

    showScoreBreakdown(pair) {
        const modal = document.createElement('div');
        modal.className = 'modal score-modal';
        modal.style.display = 'block';
        
        modal.innerHTML = `
            <div class="modal-content">
                <div class="modal-header">
                    <h3>${pair.symbol} - Score Breakdown</h3>
                    <button class="modal-close">&times;</button>
                </div>
                <div class="modal-body">
                    <div class="score-breakdown">
                        <div class="total-score">
                            <div class="score-circle">
                                <span class="score-value">${pair.score || 0}</span>
                                <span class="score-label">Total Score</span>
                            </div>
                        </div>
                        
                        <div class="factor-scores">
                            <div class="factor-item">
                                <div class="factor-label">Liquidity & Market Quality (20%)</div>
                                <div class="factor-bar">
                                    <div class="factor-fill" style="width: ${(pair.liquidity_score || 0) * 5}%"></div>
                                    <span class="factor-value">${pair.liquidity_score || 0}/20</span>
                                </div>
                            </div>
                            
                            <div class="factor-item">
                                <div class="factor-label">Volatility & Trend (20%)</div>
                                <div class="factor-bar">
                                    <div class="factor-fill" style="width: ${(pair.volatility_score || 0) * 5}%"></div>
                                    <span class="factor-value">${pair.volatility_score || 0}/20</span>
                                </div>
                            </div>
                            
                            <div class="factor-item">
                                <div class="factor-label">Momentum vs BTC/ETH (20%)</div>
                                <div class="factor-bar">
                                    <div class="factor-fill" style="width: ${(pair.momentum_score || 0) * 5}%"></div>
                                    <span class="factor-value">${pair.momentum_score || 0}/20</span>
                                </div>
                            </div>
                            
                            <div class="factor-item">
                                <div class="factor-label">Correlation & Diversification (15%)</div>
                                <div class="factor-bar">
                                    <div class="factor-fill" style="width: ${(pair.correlation_score || 0) * 100/15}%"></div>
                                    <span class="factor-value">${pair.correlation_score || 0}/15</span>
                                </div>
                            </div>
                            
                            <div class="factor-item">
                                <div class="factor-label">Sentiment & News (15%)</div>
                                <div class="factor-bar">
                                    <div class="factor-fill" style="width: ${(pair.sentiment_score || 0) * 100/15}%"></div>
                                    <span class="factor-value">${pair.sentiment_score || 0}/15</span>
                                </div>
                            </div>
                            
                            <div class="factor-item">
                                <div class="factor-label">Technical Health (10%)</div>
                                <div class="factor-bar">
                                    <div class="factor-fill" style="width: ${(pair.technical_score || 0) * 10}%"></div>
                                    <span class="factor-value">${pair.technical_score || 0}/10</span>
                                </div>
                            </div>
                        </div>
                        
                        <div class="pair-summary">
                            <h4>Market Regime: ${pair.regime || 'ranging'}</h4>
                            <div class="summary-metrics">
                                <div>Volume: $${(pair.volume24h / 1000000).toFixed(1)}M</div>
                                <div>ATR: ${pair.atr_pct || 0}%</div>
                                <div>Spread: ${pair.spread_bps || 0}bp</div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        `;
        
        document.body.appendChild(modal);
        
        // Close modal handlers
        const closeBtn = modal.querySelector('.modal-close');
        closeBtn.addEventListener('click', () => {
            document.body.removeChild(modal);
        });
        
        modal.addEventListener('click', (e) => {
            if (e.target === modal) {
                document.body.removeChild(modal);
            }
        });
    }

    destroy() {
        this.stopPeriodicUpdates();
    }
}

// Global functions for HTML onclick handlers
function startTrading(assetType) {
    if (window.dashboard) {
        window.dashboard.startTrading(assetType);
    }
}

function stopTrading(assetType) {
    if (window.dashboard) {
        window.dashboard.stopTrading(assetType);
    }
}

// Initialize dashboard when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.dashboard = new TradingBotDashboard();
});

// Handle page visibility changes to pause/resume updates
document.addEventListener('visibilitychange', () => {
    if (window.dashboard) {
        if (document.hidden) {
            window.dashboard.stopPeriodicUpdates();
        } else {
            window.dashboard.startPeriodicUpdates();
        }
    }
});