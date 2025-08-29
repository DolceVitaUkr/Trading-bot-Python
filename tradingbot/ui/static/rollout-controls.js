/**
 * Rollout Controls and Safe Reset Management
 * Handles per-asset × mode rollout toggles and safe reset operations
 */

class RolloutManager {
    constructor() {
        this.baseUrl = '';
        this.drawerOpen = false;
        this.assets = ['crypto', 'futures', 'forex', 'forex_options'];
        this.modes = ['paper', 'live'];
        this.blockers = new Map();
        
        this.init();
    }
    
    init() {
        this.createRolloutDrawer();
        this.bindGlobalEvents();
        this.loadRolloutStatus();
    }
    
    createRolloutDrawer() {
        const drawer = document.createElement('div');
        drawer.className = 'rollout-drawer';
        drawer.id = 'rolloutDrawer';
        drawer.innerHTML = `
            <div class="rollout-drawer-header">
                <h2 class="rollout-drawer-title">Rollout Controls</h2>
                <button class="btn btn-secondary" onclick="rolloutManager.toggleDrawer()">
                    <i class="fas fa-times"></i>
                </button>
            </div>
            <div class="rollout-drawer-content">
                <div class="rollout-info">
                    <p class="text-muted">Control paper and live trading rollout per asset. Live trading requires validation and broker connectivity.</p>
                </div>
                <div class="rollout-asset-list" id="rolloutAssetList">
                    <!-- Dynamic content -->
                </div>
            </div>
        `;
        document.body.appendChild(drawer);
    }
    
    bindGlobalEvents() {
        // Bind rollout toggle buttons in asset panels
        this.assets.forEach(asset => {
            // Paper toggle
            const paperToggle = document.querySelector(`#${asset}-rollout-paper`);
            if (paperToggle) {
                paperToggle.addEventListener('change', (e) => this.handleToggle(asset, 'paper', e.target.checked));
            }
            
            // Live toggle
            const liveToggle = document.querySelector(`#${asset}-rollout-live`);
            if (liveToggle) {
                liveToggle.addEventListener('change', (e) => this.handleToggle(asset, 'live', e.target.checked));
            }
        });
    }
    
    async loadRolloutStatus() {
        try {
            const response = await fetch(`${this.baseUrl}/rollout/status`);
            if (!response.ok) throw new Error('Failed to load rollout status');
            
            const data = await response.json();
            this.updateRolloutUI(data);
        } catch (error) {
            console.error('Failed to load rollout status:', error);
            this.showNotification('Failed to load rollout status', 'error');
        }
    }
    
    updateRolloutUI(statusData) {
        const assetList = document.getElementById('rolloutAssetList');
        if (!assetList) return;
        
        assetList.innerHTML = '';
        
        this.assets.forEach(asset => {
            const assetStatus = statusData[asset] || {};
            const paperStatus = assetStatus.paper || { enabled: false, blockers: [] };
            const liveStatus = assetStatus.live || { enabled: false, blockers: [] };
            
            const assetItem = document.createElement('div');
            assetItem.className = 'rollout-asset-item';
            assetItem.innerHTML = `
                <div class="rollout-asset-header">
                    <div class="asset-info">
                        <div class="asset-name">${this.formatAssetName(asset)}</div>
                        <div class="asset-broker text-muted">${this.getAssetBroker(asset)}</div>
                    </div>
                </div>
                
                <div class="rollout-controls-grid">
                    <div class="rollout-control">
                        <label class="rollout-label">Paper Trading</label>
                        <div class="rollout-switch">
                            <label class="rollout-toggle paper">
                                <input type="checkbox" 
                                       id="${asset}-drawer-paper" 
                                       ${paperStatus.enabled ? 'checked' : ''}
                                       onchange="rolloutManager.handleToggle('${asset}', 'paper', this.checked)">
                                <span class="rollout-toggle-slider"></span>
                            </label>
                        </div>
                        ${this.renderBlockers(paperStatus.blockers)}
                    </div>
                    
                    <div class="rollout-control">
                        <label class="rollout-label">Live Trading</label>
                        <div class="rollout-switch">
                            <label class="rollout-toggle live">
                                <input type="checkbox" 
                                       id="${asset}-drawer-live" 
                                       ${liveStatus.enabled ? 'checked' : ''}
                                       ${liveStatus.blockers.length > 0 ? 'disabled' : ''}
                                       onchange="rolloutManager.handleToggle('${asset}', 'live', this.checked)">
                                <span class="rollout-toggle-slider"></span>
                            </label>
                        </div>
                        ${this.renderBlockers(liveStatus.blockers)}
                    </div>
                </div>
            `;
            
            assetList.appendChild(assetItem);
        });
    }
    
    renderBlockers(blockers) {
        if (!blockers || blockers.length === 0) return '';
        
        return `
            <div class="blocker-list">
                ${blockers.map(blocker => `
                    <div class="status-chip ${blocker.type}">
                        <i class="fas fa-exclamation-triangle"></i>
                        ${blocker.reason}
                    </div>
                `).join('')}
            </div>
        `;
    }
    
    async handleToggle(asset, mode, enabled) {
        const endpoint = enabled ? 'enable' : 'disable';
        
        try {
            const response = await fetch(`${this.baseUrl}/rollout/${asset}/${mode}/${endpoint}`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' }
            });
            
            if (!response.ok) {
                const data = await response.json();
                throw new Error(data.reason || `Failed to ${endpoint} ${mode} for ${asset}`);
            }
            
            const result = await response.json();
            
            // Update all relevant toggles
            this.updateToggleStates(asset, mode, result.enabled);
            
            // Show success notification
            this.showNotification(
                `${this.formatAssetName(asset)} ${mode} trading ${result.enabled ? 'enabled' : 'disabled'}`,
                'success'
            );
            
            // Log to activity
            if (window.dashboard) {
                window.dashboard.logActivity(
                    asset.toUpperCase(),
                    `${mode} trading ${result.enabled ? 'enabled' : 'disabled'} via rollout control`,
                    'info'
                );
            }
            
            // Reload status to get updated blockers
            this.loadRolloutStatus();
            
        } catch (error) {
            console.error('Rollout toggle failed:', error);
            this.showNotification(error.message, 'error');
            
            // Revert toggle states
            this.updateToggleStates(asset, mode, !enabled);
        }
    }
    
    updateToggleStates(asset, mode, enabled) {
        // Update main panel toggle
        const mainToggle = document.querySelector(`#${asset}-rollout-${mode}`);
        if (mainToggle) mainToggle.checked = enabled;
        
        // Update drawer toggle
        const drawerToggle = document.querySelector(`#${asset}-drawer-${mode}`);
        if (drawerToggle) drawerToggle.checked = enabled;
    }
    
    toggleDrawer() {
        const drawer = document.getElementById('rolloutDrawer');
        if (!drawer) return;
        
        this.drawerOpen = !this.drawerOpen;
        
        if (this.drawerOpen) {
            drawer.classList.add('open');
            this.loadRolloutStatus(); // Refresh on open
        } else {
            drawer.classList.remove('open');
        }
    }
    
    formatAssetName(asset) {
        const names = {
            'crypto': 'Crypto Spot',
            'futures': 'Crypto Futures',
            'forex': 'Forex Spot',
            'forex_options': 'Forex Options'
        };
        return names[asset] || asset;
    }
    
    getAssetBroker(asset) {
        const brokers = {
            'crypto': 'Bybit',
            'futures': 'Bybit',
            'forex': 'IBKR',
            'forex_options': 'IBKR'
        };
        return brokers[asset] || 'Unknown';
    }
    
    showNotification(message, type = 'info') {
        const notification = document.createElement('div');
        notification.className = `notification ${type}`;
        notification.innerHTML = `
            <i class="fas fa-${type === 'success' ? 'check-circle' : type === 'error' ? 'exclamation-circle' : 'info-circle'}"></i>
            <span>${message}</span>
        `;
        
        document.body.appendChild(notification);
        
        // Trigger show animation
        setTimeout(() => notification.classList.add('show'), 10);
        
        // Auto remove
        setTimeout(() => {
            notification.classList.remove('show');
            setTimeout(() => notification.remove(), 300);
        }, 5000);
    }
}

/**
 * Safe Reset Dialog Manager
 * Handles paper history and model reset operations with safety checks
 */
class SafeResetManager {
    constructor() {
        this.baseUrl = '';
        this.activeDialog = null;
        
        this.init();
    }
    
    init() {
        this.createDialogTemplates();
    }
    
    createDialogTemplates() {
        // Create container for dialogs
        const container = document.createElement('div');
        container.id = 'safeResetDialogs';
        document.body.appendChild(container);
    }
    
    openHistoryResetDialog(asset) {
        const dialog = this.createDialog('history', asset);
        document.getElementById('safeResetDialogs').appendChild(dialog);
        
        // Initialize dialog
        this.activeDialog = {
            type: 'history',
            asset: asset,
            element: dialog
        };
        
        // Load preview data
        this.loadHistoryPreview(asset);
        
        // Show dialog
        setTimeout(() => dialog.classList.add('show'), 10);
    }
    
    openModelResetDialog(asset) {
        const dialog = this.createDialog('model', asset);
        document.getElementById('safeResetDialogs').appendChild(dialog);
        
        // Initialize dialog
        this.activeDialog = {
            type: 'model',
            asset: asset,
            element: dialog
        };
        
        // Load preview data
        this.loadModelPreview(asset);
        
        // Show dialog
        setTimeout(() => dialog.classList.add('show'), 10);
    }
    
    createDialog(type, asset) {
        const isHistory = type === 'history';
        const confirmText = isHistory ? `RESET ${asset.toUpperCase()}` : `WIPE MODELS ${asset.toUpperCase()}`;
        
        const dialog = document.createElement('div');
        dialog.className = 'modal-backdrop';
        dialog.innerHTML = `
            <div class="modal-dialog">
                <div class="modal-header">
                    <h3 class="modal-title">
                        ${isHistory ? 'Reset Paper History' : 'Reset Training Model'} - ${this.formatAssetName(asset)}
                    </h3>
                </div>
                
                <div class="modal-body">
                    <div class="preview-card">
                        <h4 class="preview-header">
                            <i class="fas fa-exclamation-triangle text-warn"></i>
                            What will be deleted:
                        </h4>
                        <div id="${type}-preview-content" class="preview-content">
                            <div class="skeleton" style="height: 100px;"></div>
                        </div>
                    </div>
                    
                    <div class="reset-options">
                        <label class="option-label">
                            <input type="checkbox" id="${type}-backup-option" checked>
                            <span>Create backup before reset</span>
                        </label>
                        
                        ${isHistory ? `
                        <label class="option-label">
                            <input type="checkbox" id="${type}-quarantine-option">
                            <span>Quarantine corrupted samples</span>
                        </label>
                        ` : ''}
                    </div>
                    
                    <div class="type-confirm">
                        <label class="type-confirm-label">
                            Type <strong>${confirmText}</strong> to confirm:
                        </label>
                        <input type="text" 
                               id="${type}-confirm-input" 
                               class="type-confirm-input"
                               placeholder="Enter confirmation text"
                               onkeyup="safeResetManager.checkConfirmation('${type}')">
                        <div class="type-confirm-hint">
                            This action cannot be undone without a backup
                        </div>
                    </div>
                </div>
                
                <div class="modal-footer">
                    <button class="btn btn-secondary" onclick="safeResetManager.closeDialog()">
                        Cancel
                    </button>
                    <button class="btn btn-primary" onclick="safeResetManager.dryRun('${type}', '${asset}')">
                        <i class="fas fa-eye"></i>
                        Preview
                    </button>
                    <button class="btn btn-danger" 
                            id="${type}-reset-btn"
                            onclick="safeResetManager.executeReset('${type}', '${asset}')"
                            disabled>
                        <i class="fas fa-trash"></i>
                        Reset ${isHistory ? 'History' : 'Model'}
                    </button>
                </div>
            </div>
        `;
        
        return dialog;
    }
    
    async loadHistoryPreview(asset) {
        try {
            const response = await fetch(`${this.baseUrl}/maintenance/preview?asset=${asset}&type=paper`);
            const data = await response.json();
            
            const content = document.getElementById('history-preview-content');
            if (content) {
                content.innerHTML = `
                    <ul class="preview-list">
                        <li class="preview-item">
                            <span class="preview-label">Paper trades</span>
                            <span class="preview-value">${data.trade_count || 0}</span>
                        </li>
                        <li class="preview-item">
                            <span class="preview-label">Paper positions</span>
                            <span class="preview-value">${data.position_count || 0}</span>
                        </li>
                        <li class="preview-item">
                            <span class="preview-label">Paper epochs</span>
                            <span class="preview-value">${data.epoch_count || 0}</span>
                        </li>
                        <li class="preview-item">
                            <span class="preview-label">Metrics data</span>
                            <span class="preview-value">${this.formatBytes(data.metrics_size || 0)}</span>
                        </li>
                        <li class="preview-item">
                            <span class="preview-label">Total size</span>
                            <span class="preview-value">${this.formatBytes(data.total_size || 0)}</span>
                        </li>
                    </ul>
                `;
            }
        } catch (error) {
            console.error('Failed to load history preview:', error);
        }
    }
    
    async loadModelPreview(asset) {
        try {
            const response = await fetch(`${this.baseUrl}/maintenance/preview?asset=${asset}&type=model`);
            const data = await response.json();
            
            const content = document.getElementById('model-preview-content');
            if (content) {
                content.innerHTML = `
                    <ul class="preview-list">
                        <li class="preview-item">
                            <span class="preview-label">Model checkpoints</span>
                            <span class="preview-value">${data.checkpoint_count || 0}</span>
                        </li>
                        <li class="preview-item">
                            <span class="preview-label">Replay buffers</span>
                            <span class="preview-value">${data.buffer_count || 0}</span>
                        </li>
                        <li class="preview-item">
                            <span class="preview-label">Feature store</span>
                            <span class="preview-value">${this.formatBytes(data.feature_size || 0)}</span>
                        </li>
                        <li class="preview-item">
                            <span class="preview-label">Training logs</span>
                            <span class="preview-value">${this.formatBytes(data.log_size || 0)}</span>
                        </li>
                        <li class="preview-item">
                            <span class="preview-label">Total size</span>
                            <span class="preview-value">${this.formatBytes(data.total_size || 0)}</span>
                        </li>
                    </ul>
                `;
            }
        } catch (error) {
            console.error('Failed to load model preview:', error);
        }
    }
    
    checkConfirmation(type) {
        if (!this.activeDialog) return;
        
        const input = document.getElementById(`${type}-confirm-input`);
        const button = document.getElementById(`${type}-reset-btn`);
        
        if (!input || !button) return;
        
        const expectedText = type === 'history' ? 
            `RESET ${this.activeDialog.asset.toUpperCase()}` : 
            `WIPE MODELS ${this.activeDialog.asset.toUpperCase()}`;
        
        button.disabled = input.value !== expectedText;
    }
    
    async dryRun(type, asset) {
        const endpoint = type === 'history' ? 'paper_reset' : 'model_reset';
        const backupOption = document.getElementById(`${type}-backup-option`);
        const quarantineOption = document.getElementById(`${type}-quarantine-option`);
        
        const payload = {
            asset: asset,
            dry_run: true,
            backup: backupOption ? backupOption.checked : true,
            reason: `User initiated ${type} reset`
        };
        
        if (type === 'history' && quarantineOption) {
            payload.quarantine = quarantineOption.checked;
        }
        
        try {
            const response = await fetch(`${this.baseUrl}/maintenance/${endpoint}`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });
            
            const result = await response.json();
            
            this.showNotification(
                `Dry run complete. ${result.affected_count || 0} items would be deleted.`,
                'info'
            );
            
        } catch (error) {
            console.error('Dry run failed:', error);
            this.showNotification('Dry run failed', 'error');
        }
    }
    
    async executeReset(type, asset) {
        const endpoint = type === 'history' ? 'paper_reset' : 'model_reset';
        const backupOption = document.getElementById(`${type}-backup-option`);
        const quarantineOption = document.getElementById(`${type}-quarantine-option`);
        
        // Double check confirmation
        const confirmInput = document.getElementById(`${type}-confirm-input`);
        const expectedText = type === 'history' ? 
            `RESET ${asset.toUpperCase()}` : 
            `WIPE MODELS ${asset.toUpperCase()}`;
        
        if (confirmInput.value !== expectedText) {
            this.showNotification('Please type the confirmation text exactly', 'error');
            return;
        }
        
        const payload = {
            asset: asset,
            dry_run: false,
            backup: backupOption ? backupOption.checked : true,
            reason: `User initiated ${type} reset`
        };
        
        if (type === 'history' && quarantineOption) {
            payload.quarantine = quarantineOption.checked;
        }
        
        try {
            // Show loading state
            const resetBtn = document.getElementById(`${type}-reset-btn`);
            if (resetBtn) {
                resetBtn.disabled = true;
                resetBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Resetting...';
            }
            
            const response = await fetch(`${this.baseUrl}/maintenance/${endpoint}`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });
            
            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.detail || 'Reset failed');
            }
            
            const result = await response.json();
            
            // Success
            this.showNotification(
                `${type === 'history' ? 'Paper history' : 'Training model'} reset successfully`,
                'success'
            );
            
            // Log activity
            if (window.dashboard) {
                window.dashboard.logActivity(
                    asset.toUpperCase(),
                    `${type === 'history' ? 'Paper history' : 'Training model'} reset (${result.deleted_count || 0} items)`,
                    'warning'
                );
            }
            
            // Close dialog
            this.closeDialog();
            
            // Refresh dashboard
            if (window.dashboard) {
                window.dashboard.updateAsset(asset);
            }
            
        } catch (error) {
            console.error('Reset failed:', error);
            this.showNotification(error.message, 'error');
            
            // Reset button state
            const resetBtn = document.getElementById(`${type}-reset-btn`);
            if (resetBtn) {
                resetBtn.disabled = false;
                resetBtn.innerHTML = `<i class="fas fa-trash"></i> Reset ${type === 'history' ? 'History' : 'Model'}`;
            }
        }
    }
    
    closeDialog() {
        if (!this.activeDialog) return;
        
        const dialog = this.activeDialog.element;
        dialog.classList.remove('show');
        
        setTimeout(() => {
            dialog.remove();
            this.activeDialog = null;
        }, 300);
    }
    
    formatAssetName(asset) {
        const names = {
            'crypto': 'Crypto Spot',
            'futures': 'Crypto Futures',
            'forex': 'Forex Spot',
            'forex_options': 'Forex Options'
        };
        return names[asset] || asset;
    }
    
    formatBytes(bytes) {
        if (bytes === 0) return '0 B';
        const k = 1024;
        const sizes = ['B', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }
    
    showNotification(message, type = 'info') {
        const notification = document.createElement('div');
        notification.className = `notification ${type}`;
        notification.innerHTML = `
            <i class="fas fa-${type === 'success' ? 'check-circle' : type === 'error' ? 'exclamation-circle' : 'info-circle'}"></i>
            <span>${message}</span>
        `;
        
        document.body.appendChild(notification);
        
        // Trigger show animation
        setTimeout(() => notification.classList.add('show'), 10);
        
        // Auto remove
        setTimeout(() => {
            notification.classList.remove('show');
            setTimeout(() => notification.remove(), 300);
        }, 5000);
    }
}

// Initialize managers
const rolloutManager = new RolloutManager();
const safeResetManager = new SafeResetManager();

// Global functions for HTML
window.rolloutManager = rolloutManager;
window.safeResetManager = safeResetManager;