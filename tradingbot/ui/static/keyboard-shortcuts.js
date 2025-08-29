/**
 * Keyboard Shortcuts Handler
 * Implements global keyboard shortcuts for navigation and actions
 */

class KeyboardShortcuts {
    constructor() {
        this.shortcuts = new Map();
        this.searchActive = false;
        this.lastKeyTime = 0;
        this.doubleKeyThreshold = 500; // ms for double key presses
        
        this.init();
    }
    
    init() {
        this.registerShortcuts();
        this.bindEventListeners();
        this.createHelpDialog();
    }
    
    registerShortcuts() {
        // Navigation shortcuts
        this.shortcuts.set('g d', { 
            action: () => this.navigateToTab('dashboard'),
            description: 'Go to Dashboard',
            category: 'Navigation'
        });
        
        this.shortcuts.set('g s', { 
            action: () => this.navigateToTab('strategy'),
            description: 'Go to Strategy',
            category: 'Navigation'
        });
        
        this.shortcuts.set('g h', { 
            action: () => this.navigateToTab('history'),
            description: 'Go to History',
            category: 'Navigation'
        });
        
        this.shortcuts.set('g e', { 
            action: () => this.navigateToTab('settings'),
            description: 'Go to Settings',
            category: 'Navigation'
        });
        
        this.shortcuts.set('g l', { 
            action: () => this.navigateToTab('logs'),
            description: 'Go to Logs',
            category: 'Navigation'
        });
        
        // Action shortcuts
        this.shortcuts.set('/', { 
            action: () => this.activateSearch(),
            description: 'Search',
            category: 'Actions',
            preventDefault: true
        });
        
        this.shortcuts.set('r r', { 
            action: () => this.openResetDialog(),
            description: 'Open Reset Dialog (admin only)',
            category: 'Actions',
            requiresAdmin: true
        });
        
        this.shortcuts.set('r d', { 
            action: () => rolloutManager.toggleDrawer(),
            description: 'Toggle Rollout Drawer',
            category: 'Actions'
        });
        
        this.shortcuts.set('?', { 
            action: () => this.showHelp(),
            description: 'Show Keyboard Shortcuts',
            category: 'Help'
        });
        
        this.shortcuts.set('Escape', { 
            action: () => this.escapeAction(),
            description: 'Close dialogs/Exit search',
            category: 'Actions'
        });
        
        // Asset-specific shortcuts (1-4 for assets)
        this.shortcuts.set('1', { 
            action: () => this.focusAsset('crypto'),
            description: 'Focus Crypto Spot',
            category: 'Assets'
        });
        
        this.shortcuts.set('2', { 
            action: () => this.focusAsset('futures'),
            description: 'Focus Crypto Futures',
            category: 'Assets'
        });
        
        this.shortcuts.set('3', { 
            action: () => this.focusAsset('forex'),
            description: 'Focus Forex Spot',
            category: 'Assets'
        });
        
        this.shortcuts.set('4', { 
            action: () => this.focusAsset('forex_options'),
            description: 'Focus Forex Options',
            category: 'Assets'
        });
        
        // Trading controls
        this.shortcuts.set('p', { 
            action: () => this.togglePaperTrading(),
            description: 'Toggle Paper Trading (focused asset)',
            category: 'Trading'
        });
        
        this.shortcuts.set('l', { 
            action: () => this.toggleLiveTrading(),
            description: 'Toggle Live Trading (focused asset)',
            category: 'Trading'
        });
        
        this.shortcuts.set('k', { 
            action: () => this.killFocusedAsset(),
            description: 'Kill focused asset trading',
            category: 'Trading'
        });
        
        this.shortcuts.set('Shift+K', { 
            action: () => this.emergencyStopAll(),
            description: 'Emergency Stop All',
            category: 'Trading',
            requiresConfirm: true
        });
    }
    
    bindEventListeners() {
        document.addEventListener('keydown', (e) => this.handleKeyDown(e));
    }
    
    handleKeyDown(e) {
        // Ignore if typing in input/textarea
        if (this.isTyping()) return;
        
        // Build key combination
        let key = '';
        if (e.ctrlKey) key += 'Ctrl+';
        if (e.altKey) key += 'Alt+';
        if (e.shiftKey) key += 'Shift+';
        if (e.metaKey) key += 'Meta+';
        
        // Add the actual key
        if (e.key === ' ') {
            key += 'Space';
        } else if (e.key.length === 1) {
            key += e.key.toLowerCase();
        } else {
            key += e.key;
        }
        
        // Check for double key sequences (like 'g d', 'r r')
        const now = Date.now();
        if (now - this.lastKeyTime < this.doubleKeyThreshold) {
            const doubleKey = `${this.lastKey} ${key}`;
            const shortcut = this.shortcuts.get(doubleKey);
            
            if (shortcut) {
                this.executeShortcut(shortcut, e);
                this.lastKey = null;
                return;
            }
        }
        
        // Check single key shortcut
        const shortcut = this.shortcuts.get(key);
        if (shortcut) {
            this.executeShortcut(shortcut, e);
        }
        
        // Store for potential double key
        this.lastKey = key;
        this.lastKeyTime = now;
    }
    
    executeShortcut(shortcut, event) {
        // Check admin requirement
        if (shortcut.requiresAdmin && !this.isAdmin()) {
            this.showNotification('Admin access required', 'warning');
            return;
        }
        
        // Prevent default if specified
        if (shortcut.preventDefault) {
            event.preventDefault();
        }
        
        // Require confirmation if specified
        if (shortcut.requiresConfirm) {
            if (!confirm('Are you sure you want to perform this action?')) {
                return;
            }
        }
        
        // Execute the action
        shortcut.action();
    }
    
    isTyping() {
        const activeElement = document.activeElement;
        const typingElements = ['INPUT', 'TEXTAREA', 'SELECT'];
        return typingElements.includes(activeElement.tagName) || activeElement.contentEditable === 'true';
    }
    
    isAdmin() {
        // Check if user has admin privileges
        // This would typically check against user session/role
        return true; // For demo purposes
    }
    
    navigateToTab(tab) {
        const tabElement = document.querySelector(`[data-tab="${tab}"]`);
        if (tabElement) {
            tabElement.click();
        }
    }
    
    activateSearch() {
        const searchInput = document.getElementById('globalSearch');
        if (searchInput) {
            searchInput.focus();
            this.searchActive = true;
        }
    }
    
    openResetDialog() {
        // Show asset selection dialog
        const dialog = document.createElement('div');
        dialog.className = 'modal-backdrop show';
        dialog.innerHTML = `
            <div class="modal-dialog">
                <div class="modal-header">
                    <h3 class="modal-title">Select Reset Type</h3>
                </div>
                <div class="modal-body">
                    <div class="reset-options-grid">
                        ${['crypto', 'futures', 'forex', 'forex_options'].map(asset => `
                            <div class="reset-option-card">
                                <h4>${this.formatAssetName(asset)}</h4>
                                <button class="btn btn-secondary" 
                                        onclick="safeResetManager.openHistoryResetDialog('${asset}'); 
                                                 keyboardShortcuts.closeDialog(this)">
                                    <i class="fas fa-history"></i>
                                    Reset Paper History
                                </button>
                                <button class="btn btn-secondary" 
                                        onclick="safeResetManager.openModelResetDialog('${asset}'); 
                                                 keyboardShortcuts.closeDialog(this)">
                                    <i class="fas fa-brain"></i>
                                    Reset Training Model
                                </button>
                            </div>
                        `).join('')}
                    </div>
                </div>
                <div class="modal-footer">
                    <button class="btn btn-secondary" onclick="keyboardShortcuts.closeDialog(this)">
                        Cancel
                    </button>
                </div>
            </div>
        `;
        
        document.body.appendChild(dialog);
    }
    
    closeDialog(button) {
        const dialog = button.closest('.modal-backdrop');
        if (dialog) {
            dialog.classList.remove('show');
            setTimeout(() => dialog.remove(), 300);
        }
    }
    
    escapeAction() {
        // Close any open modals
        document.querySelectorAll('.modal-backdrop.show').forEach(modal => {
            modal.classList.remove('show');
            setTimeout(() => modal.remove(), 300);
        });
        
        // Close rollout drawer
        if (rolloutManager && rolloutManager.drawerOpen) {
            rolloutManager.toggleDrawer();
        }
        
        // Exit search
        if (this.searchActive) {
            const searchInput = document.getElementById('globalSearch');
            if (searchInput) {
                searchInput.blur();
                searchInput.value = '';
                this.searchActive = false;
            }
        }
    }
    
    focusAsset(asset) {
        const assetPanel = document.querySelector(`[data-asset="${asset}"]`);
        if (assetPanel) {
            assetPanel.scrollIntoView({ behavior: 'smooth', block: 'center' });
            assetPanel.classList.add('focused');
            
            // Remove focus after animation
            setTimeout(() => assetPanel.classList.remove('focused'), 1500);
            
            this.focusedAsset = asset;
        }
    }
    
    togglePaperTrading() {
        if (!this.focusedAsset) {
            this.showNotification('Focus an asset first (press 1-4)', 'info');
            return;
        }
        
        window.toggleTrading(this.focusedAsset, 'paper');
    }
    
    toggleLiveTrading() {
        if (!this.focusedAsset) {
            this.showNotification('Focus an asset first (press 1-4)', 'info');
            return;
        }
        
        window.toggleTrading(this.focusedAsset, 'live');
    }
    
    killFocusedAsset() {
        if (!this.focusedAsset) {
            this.showNotification('Focus an asset first (press 1-4)', 'info');
            return;
        }
        
        window.killAsset(this.focusedAsset);
    }
    
    emergencyStopAll() {
        if (window.dashboard) {
            window.dashboard.emergencyStopAll();
        }
    }
    
    createHelpDialog() {
        const helpDialog = document.createElement('div');
        helpDialog.id = 'keyboardHelpDialog';
        helpDialog.className = 'modal-backdrop';
        helpDialog.innerHTML = `
            <div class="modal-dialog modal-lg">
                <div class="modal-header">
                    <h3 class="modal-title">Keyboard Shortcuts</h3>
                </div>
                <div class="modal-body">
                    ${this.generateHelpContent()}
                </div>
                <div class="modal-footer">
                    <button class="btn btn-primary" onclick="keyboardShortcuts.hideHelp()">
                        Close (Esc)
                    </button>
                </div>
            </div>
        `;
        
        document.body.appendChild(helpDialog);
    }
    
    generateHelpContent() {
        const categories = {};
        
        // Group shortcuts by category
        this.shortcuts.forEach((shortcut, key) => {
            const category = shortcut.category || 'Other';
            if (!categories[category]) {
                categories[category] = [];
            }
            categories[category].push({ key, ...shortcut });
        });
        
        // Generate HTML
        return Object.entries(categories).map(([category, shortcuts]) => `
            <div class="shortcut-category">
                <h4 class="category-title">${category}</h4>
                <div class="shortcut-list">
                    ${shortcuts.map(shortcut => `
                        <div class="shortcut-item">
                            <kbd class="shortcut-key">${shortcut.key}</kbd>
                            <span class="shortcut-description">${shortcut.description}</span>
                        </div>
                    `).join('')}
                </div>
            </div>
        `).join('');
    }
    
    showHelp() {
        const dialog = document.getElementById('keyboardHelpDialog');
        if (dialog) {
            dialog.classList.add('show');
        }
    }
    
    hideHelp() {
        const dialog = document.getElementById('keyboardHelpDialog');
        if (dialog) {
            dialog.classList.remove('show');
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
    
    showNotification(message, type = 'info') {
        const notification = document.createElement('div');
        notification.className = `notification ${type}`;
        notification.innerHTML = `
            <i class="fas fa-${type === 'success' ? 'check-circle' : 
                            type === 'error' ? 'exclamation-circle' : 
                            type === 'warning' ? 'exclamation-triangle' : 
                            'info-circle'}"></i>
            <span>${message}</span>
        `;
        
        document.body.appendChild(notification);
        
        setTimeout(() => notification.classList.add('show'), 10);
        
        setTimeout(() => {
            notification.classList.remove('show');
            setTimeout(() => notification.remove(), 300);
        }, 3000);
    }
}

// Style additions for help dialog and shortcuts
const style = document.createElement('style');
style.textContent = `
.modal-lg {
    max-width: 800px;
}

.shortcut-category {
    margin-bottom: 24px;
}

.category-title {
    font-size: 16px;
    font-weight: 600;
    color: var(--text);
    margin-bottom: 12px;
    padding-bottom: 8px;
    border-bottom: 1px solid var(--line);
}

.shortcut-list {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
    gap: 8px;
}

.shortcut-item {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 8px 12px;
    background: rgba(255, 255, 255, 0.03);
    border-radius: 8px;
}

.shortcut-key {
    min-width: 60px;
}

.shortcut-description {
    color: var(--muted);
    font-size: 13px;
}

.asset-panel.focused {
    animation: focusPulse 1.5s ease;
}

@keyframes focusPulse {
    0%, 100% {
        box-shadow: var(--shadow-soft);
    }
    50% {
        box-shadow: 0 0 0 3px var(--cobalt), var(--shadow-hover);
    }
}

.reset-options-grid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 16px;
}

.reset-option-card {
    padding: 16px;
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid var(--line);
    border-radius: 12px;
}

.reset-option-card h4 {
    font-size: 14px;
    margin-bottom: 12px;
    color: var(--text);
}

.reset-option-card button {
    width: 100%;
    margin-bottom: 8px;
}

.reset-option-card button:last-child {
    margin-bottom: 0;
}
`;
document.head.appendChild(style);

// Initialize keyboard shortcuts
const keyboardShortcuts = new KeyboardShortcuts();
window.keyboardShortcuts = keyboardShortcuts;