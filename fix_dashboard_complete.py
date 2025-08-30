"""
Completely rebuild the dashboard HTML with proper structure
"""

# Read the current HTML
with open('tradingbot/ui/templates/dashboard.html', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Find where the asset cards start
asset_start = -1
for i, line in enumerate(lines):
    if '<div class="assets-container">' in line:
        asset_start = i
        break

if asset_start == -1:
    print("Could not find assets container")
    exit(1)

# Keep everything before assets container
new_html = lines[:asset_start+1]

# Asset configurations
assets = [
    {'id': 'crypto', 'name': 'Crypto Spot', 'broker': 'Bybit', 'icon': 'fab fa-bitcoin'},
    {'id': 'futures', 'name': 'Crypto Futures', 'broker': 'Bybit', 'icon': 'fas fa-chart-area'},
    {'id': 'forex', 'name': 'Forex', 'broker': 'IBKR', 'icon': 'fas fa-dollar-sign'},
    {'id': 'forex_options', 'name': 'Forex Options', 'broker': 'IBKR', 'icon': 'fas fa-coins'}
]

# Generate each asset card
for asset in assets:
    asset_html = f'''            <!-- {asset['name']} -->
            <div class="asset-card" data-asset="{asset['id']}">
                <div class="asset-header">
                    <div class="asset-info">
                        <div class="asset-icon {asset['id']}">
                            <i class="{asset['icon']}"></i>
                        </div>
                        <div class="asset-details">
                            <h3>{asset['name']}</h3>
                            <span class="broker">{asset['broker']}</span>
                        </div>
                    </div>
                    <div class="connection-status offline" id="{asset['id']}-connection-status">
                        <div class="status-dot"></div>
                        <span>Offline</span>
                    </div>
                </div>

                <!-- Paper Trading Section -->
                <div class="trading-section paper" id="{asset['id']}-paper-section">
                    <div class="section-header">
                        <h4><i class="fas fa-flask"></i> Strategy Development</h4>
                        <div class="status-controls">
                            <div class="trading-status" id="{asset['id']}-paper-status">
                                <span class="status-indicator idle"></span>
                                <span class="status-text">Idle</span>
                            </div>
                            <div class="controls">
                                <div class="trading-toggle">
                                    <label class="toggle-switch">
                                        <input type="checkbox" id="{asset['id']}-paper-toggle" onchange="handleTradingToggle('{asset['id']}', 'paper', this.checked)">
                                        <span class="toggle-slider"></span>
                                    </label>
                                    <span class="toggle-label">Paper Trading</span>
                                </div>
                                <button class="btn btn-kill" onclick="killAsset('{asset['id']}')">Kill</button>
                            </div>
                        </div>
                    </div>
                    <div class="trading-content">
                        <div class="balance-section">
                            <div class="balance-info">
                                <div class="balance-value" id="{asset['id']}-paper-balance">$0.00</div>
                                <div class="balance-change" id="{asset['id']}-paper-change">+$0.00 (0.00%)</div>
                            </div>
                            <div class="balance-chart">
                                <canvas id="{asset['id']}-paper-chart" width="120" height="60"></canvas>
                            </div>
                        </div>
                        <div class="metrics-grid">
                            <div class="metric">
                                <span class="metric-label">Positions</span>
                                <span class="metric-value" id="{asset['id']}-paper-positions">0</span>
                            </div>
                            <div class="metric">
                                <span class="metric-label">Trades</span>
                                <span class="metric-value" id="{asset['id']}-paper-trades">0</span>
                            </div>
                            <div class="metric">
                                <span class="metric-label">Win Rate</span>
                                <span class="metric-value" id="{asset['id']}-paper-winrate">0%</span>
                            </div>
                            <div class="metric">
                                <span class="metric-label">Avg P&L</span>
                                <span class="metric-value" id="{asset['id']}-paper-avgpnl">$0</span>
                            </div>
                        </div>
                        
                        <!-- Active Positions -->
                        <div class="positions-section" id="{asset['id']}-paper-positions-section" style="margin-top: 10px;">
                            <div class="section-header" style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                                <h5 style="margin: 0; font-size: 14px; color: #e2e8f0;">
                                    <i class="fas fa-chart-line"></i> Active Positions
                                </h5>
                                <div class="positions-controls" style="display: flex; gap: 10px;">
                                    <input type="text" id="{asset['id']}-paper-positions-search" placeholder="Search symbol..." style="font-size: 12px; padding: 4px 8px; background: #1a202c; color: #e2e8f0; border: 1px solid #2d3748; border-radius: 4px; width: 120px;">
                                    <select id="{asset['id']}-paper-positions-sort" class="positions-sort" style="font-size: 12px; padding: 4px 8px; background: #1a202c; color: #e2e8f0; border: 1px solid #2d3748; border-radius: 4px;">
                                        <option value="symbol">Symbol</option>
                                        <option value="pnl-desc">P&L High to Low</option>
                                        <option value="pnl-asc">P&L Low to High</option>
                                        <option value="size-desc">Size Large to Small</option>
                                    </select>
                                </div>
                            </div>
                            <div class="table-wrapper" style="max-height: 120px; overflow-y: auto; background: #1a202c; border-radius: 6px;">
                                <table class="positions-table" id="{asset['id']}-paper-positions-table" style="width: 100%; font-size: 12px;">
                                    <thead style="position: sticky; top: 0; background: #2d3748; z-index: 10;">
                                        <tr>
                                            <th style="padding: 8px; text-align: left; color: #a0aec0;">Symbol</th>
                                            <th style="padding: 8px; text-align: left; color: #a0aec0;">Side</th>
                                            <th style="padding: 8px; text-align: left; color: #a0aec0;">Size</th>
                                            <th style="padding: 8px; text-align: left; color: #a0aec0;">Entry</th>
                                            <th style="padding: 8px; text-align: left; color: #a0aec0;">Current</th>
                                            <th style="padding: 8px; text-align: left; color: #a0aec0;">P&L</th>
                                            <th style="padding: 8px; text-align: left; color: #a0aec0;">TP/SL</th>
                                        </tr>
                                    </thead>
                                    <tbody id="{asset['id']}-paper-positions-tbody">
                                        <tr>
                                            <td colspan="7" style="text-align: center; padding: 20px; color: #6c757d;">No active positions</td>
                                        </tr>
                                    </tbody>
                                </table>
                            </div>
                        </div>
                        
                        <!-- Activity Log -->
                        <div class="activity-log-section" id="{asset['id']}-paper-activity-section" style="margin-top: 15px;">
                            <div class="section-header" style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                                <h5 style="margin: 0; font-size: 14px; color: #e2e8f0;">
                                    <i class="fas fa-history"></i> Activity Log
                                </h5>
                                <div class="activity-controls" style="display: flex; gap: 10px;">
                                    <select id="{asset['id']}-paper-activity-filter" class="activity-filter" style="font-size: 12px; padding: 4px 8px; background: #1a202c; color: #e2e8f0; border: 1px solid #2d3748; border-radius: 4px;">
                                        <option value="all">All Activities</option>
                                        <option value="trade">Trades Only</option>
                                        <option value="position">Position Changes</option>
                                        <option value="error">Errors Only</option>
                                    </select>
                                </div>
                            </div>
                            <div class="activity-list" id="{asset['id']}-paper-activity-list" style="max-height: 150px; overflow-y: auto; background: #1a202c; border-radius: 6px; padding: 8px;">
                                <div class="no-activity" style="text-align: center; color: #6c757d; padding: 20px; font-size: 12px;">No activity yet</div>
                            </div>
                        </div>
                        
                        <!-- Trade History -->
                        <div class="history-section" id="{asset['id']}-paper-history-section" style="margin-top: 15px;">
                            <div class="section-header" style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                                <h5 style="margin: 0; font-size: 14px; color: #e2e8f0;">
                                    <i class="fas fa-table"></i> Trade History
                                </h5>
                                <div class="history-controls" style="display: flex; gap: 10px;">
                                    <input type="text" id="{asset['id']}-paper-history-search" placeholder="Search..." style="font-size: 12px; padding: 4px 8px; background: #1a202c; color: #e2e8f0; border: 1px solid #2d3748; border-radius: 4px; width: 120px;">
                                    <select id="{asset['id']}-paper-history-sort" class="history-sort" style="font-size: 12px; padding: 4px 8px; background: #1a202c; color: #e2e8f0; border: 1px solid #2d3748; border-radius: 4px;">
                                        <option value="time-desc">Newest First</option>
                                        <option value="time-asc">Oldest First</option>
                                        <option value="pnl-desc">Highest P&L</option>
                                        <option value="pnl-asc">Lowest P&L</option>
                                    </select>
                                </div>
                            </div>
                            <div class="table-wrapper" style="max-height: 200px; overflow-y: auto; background: #1a202c; border-radius: 6px;">
                                <table class="history-table" id="{asset['id']}-paper-history-table" style="width: 100%; font-size: 12px;">
                                    <thead style="position: sticky; top: 0; background: #2d3748;">
                                        <tr>
                                            <th style="padding: 8px; text-align: left; color: #a0aec0;">Mode</th>
                                            <th style="padding: 8px; text-align: left; color: #a0aec0;">Side</th>
                                            <th style="padding: 8px; text-align: left; color: #a0aec0;">Time Open</th>
                                            <th style="padding: 8px; text-align: left; color: #a0aec0;">Size USD</th>
                                            <th style="padding: 8px; text-align: left; color: #a0aec0;">Open Price</th>
                                            <th style="padding: 8px; text-align: left; color: #a0aec0;">Time Closed</th>
                                            <th style="padding: 8px; text-align: left; color: #a0aec0;">Close Price</th>
                                            <th style="padding: 8px; text-align: left; color: #a0aec0;">P&L USD</th>
                                            <th style="padding: 8px; text-align: left; color: #a0aec0;">P&L %</th>
                                            <th style="padding: 8px; text-align: left; color: #a0aec0;">Balance</th>
                                        </tr>
                                    </thead>
                                    <tbody id="{asset['id']}-paper-history-tbody">
                                        <tr>
                                            <td colspan="10" style="text-align: center; padding: 20px; color: #6c757d;">No completed trades yet</td>
                                        </tr>
                                    </tbody>
                                </table>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- Live Trading Section -->
                <div class="trading-section live" id="{asset['id']}-live-section">
                    <div class="section-header">
                        <h4><i class="fas fa-fire"></i> Live Trading</h4>
                        <div class="status-controls">
                            <div class="trading-status" id="{asset['id']}-live-status">
                                <span class="status-indicator idle"></span>
                                <span class="status-text">Idle</span>
                            </div>
                            <div class="controls">
                                <div class="trading-toggle">
                                    <label class="toggle-switch">
                                        <input type="checkbox" id="{asset['id']}-live-toggle" onchange="handleTradingToggle('{asset['id']}', 'live', this.checked)">
                                        <span class="toggle-slider"></span>
                                    </label>
                                    <span class="toggle-label">Live Trading</span>
                                </div>
                                <button class="btn btn-kill" onclick="killAsset('{asset['id']}')">Kill</button>
                            </div>
                        </div>
                    </div>
                    <div class="trading-content">
                        <div class="balance-section">
                            <div class="balance-info">
                                <div class="balance-value" id="{asset['id']}-live-balance">$0.00</div>
                                <div class="balance-change" id="{asset['id']}-live-change">+$0.00 (0.00%)</div>
                            </div>
                            <div class="balance-chart">
                                <canvas id="{asset['id']}-live-chart" width="120" height="60"></canvas>
                            </div>
                        </div>
                        <div class="metrics-grid">
                            <div class="metric">
                                <span class="metric-label">Positions</span>
                                <span class="metric-value" id="{asset['id']}-live-positions">0</span>
                            </div>
                            <div class="metric">
                                <span class="metric-label">Trades</span>
                                <span class="metric-value" id="{asset['id']}-live-trades">0</span>
                            </div>
                            <div class="metric">
                                <span class="metric-label">Win Rate</span>
                                <span class="metric-value" id="{asset['id']}-live-winrate">0%</span>
                            </div>
                            <div class="metric">
                                <span class="metric-label">Avg P&L</span>
                                <span class="metric-value" id="{asset['id']}-live-avgpnl">$0</span>
                            </div>
                        </div>
                        
                        <!-- Active Positions -->
                        <div class="positions-section" id="{asset['id']}-live-positions-section" style="margin-top: 10px;">
                            <div class="section-header" style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                                <h5 style="margin: 0; font-size: 14px; color: #e2e8f0;">
                                    <i class="fas fa-chart-line"></i> Active Positions
                                </h5>
                                <div class="positions-controls" style="display: flex; gap: 10px;">
                                    <input type="text" id="{asset['id']}-live-positions-search" placeholder="Search symbol..." style="font-size: 12px; padding: 4px 8px; background: #1a202c; color: #e2e8f0; border: 1px solid #2d3748; border-radius: 4px; width: 120px;">
                                    <select id="{asset['id']}-live-positions-sort" class="positions-sort" style="font-size: 12px; padding: 4px 8px; background: #1a202c; color: #e2e8f0; border: 1px solid #2d3748; border-radius: 4px;">
                                        <option value="symbol">Symbol</option>
                                        <option value="pnl-desc">P&L High to Low</option>
                                        <option value="pnl-asc">P&L Low to High</option>
                                        <option value="size-desc">Size Large to Small</option>
                                    </select>
                                </div>
                            </div>
                            <div class="table-wrapper" style="max-height: 120px; overflow-y: auto; background: #1a202c; border-radius: 6px;">
                                <table class="positions-table" id="{asset['id']}-live-positions-table" style="width: 100%; font-size: 12px;">
                                    <thead style="position: sticky; top: 0; background: #2d3748; z-index: 10;">
                                        <tr>
                                            <th style="padding: 8px; text-align: left; color: #a0aec0;">Symbol</th>
                                            <th style="padding: 8px; text-align: left; color: #a0aec0;">Side</th>
                                            <th style="padding: 8px; text-align: left; color: #a0aec0;">Size</th>
                                            <th style="padding: 8px; text-align: left; color: #a0aec0;">Entry</th>
                                            <th style="padding: 8px; text-align: left; color: #a0aec0;">Current</th>
                                            <th style="padding: 8px; text-align: left; color: #a0aec0;">P&L</th>
                                            <th style="padding: 8px; text-align: left; color: #a0aec0;">TP/SL</th>
                                        </tr>
                                    </thead>
                                    <tbody id="{asset['id']}-live-positions-tbody">
                                        <tr>
                                            <td colspan="7" style="text-align: center; padding: 20px; color: #6c757d;">No active positions</td>
                                        </tr>
                                    </tbody>
                                </table>
                            </div>
                        </div>
                        
                        <!-- Activity Log -->
                        <div class="activity-log-section" id="{asset['id']}-live-activity-section" style="margin-top: 15px;">
                            <div class="section-header" style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                                <h5 style="margin: 0; font-size: 14px; color: #e2e8f0;">
                                    <i class="fas fa-history"></i> Activity Log
                                </h5>
                                <div class="activity-controls" style="display: flex; gap: 10px;">
                                    <select id="{asset['id']}-live-activity-filter" class="activity-filter" style="font-size: 12px; padding: 4px 8px; background: #1a202c; color: #e2e8f0; border: 1px solid #2d3748; border-radius: 4px;">
                                        <option value="all">All Activities</option>
                                        <option value="trade">Trades Only</option>
                                        <option value="position">Position Changes</option>
                                        <option value="error">Errors Only</option>
                                    </select>
                                </div>
                            </div>
                            <div class="activity-list" id="{asset['id']}-live-activity-list" style="max-height: 150px; overflow-y: auto; background: #1a202c; border-radius: 6px; padding: 8px;">
                                <div class="no-activity" style="text-align: center; color: #6c757d; padding: 20px; font-size: 12px;">No activity yet</div>
                            </div>
                        </div>
                        
                        <!-- Trade History -->
                        <div class="history-section" id="{asset['id']}-live-history-section" style="margin-top: 15px;">
                            <div class="section-header" style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                                <h5 style="margin: 0; font-size: 14px; color: #e2e8f0;">
                                    <i class="fas fa-table"></i> Trade History
                                </h5>
                                <div class="history-controls" style="display: flex; gap: 10px;">
                                    <input type="text" id="{asset['id']}-live-history-search" placeholder="Search..." style="font-size: 12px; padding: 4px 8px; background: #1a202c; color: #e2e8f0; border: 1px solid #2d3748; border-radius: 4px; width: 120px;">
                                    <select id="{asset['id']}-live-history-sort" class="history-sort" style="font-size: 12px; padding: 4px 8px; background: #1a202c; color: #e2e8f0; border: 1px solid #2d3748; border-radius: 4px;">
                                        <option value="time-desc">Newest First</option>
                                        <option value="time-asc">Oldest First</option>
                                        <option value="pnl-desc">Highest P&L</option>
                                        <option value="pnl-asc">Lowest P&L</option>
                                    </select>
                                </div>
                            </div>
                            <div class="table-wrapper" style="max-height: 200px; overflow-y: auto; background: #1a202c; border-radius: 6px;">
                                <table class="history-table" id="{asset['id']}-live-history-table" style="width: 100%; font-size: 12px;">
                                    <thead style="position: sticky; top: 0; background: #2d3748;">
                                        <tr>
                                            <th style="padding: 8px; text-align: left; color: #a0aec0;">Mode</th>
                                            <th style="padding: 8px; text-align: left; color: #a0aec0;">Side</th>
                                            <th style="padding: 8px; text-align: left; color: #a0aec0;">Time Open</th>
                                            <th style="padding: 8px; text-align: left; color: #a0aec0;">Size USD</th>
                                            <th style="padding: 8px; text-align: left; color: #a0aec0;">Open Price</th>
                                            <th style="padding: 8px; text-align: left; color: #a0aec0;">Time Closed</th>
                                            <th style="padding: 8px; text-align: left; color: #a0aec0;">Close Price</th>
                                            <th style="padding: 8px; text-align: left; color: #a0aec0;">P&L USD</th>
                                            <th style="padding: 8px; text-align: left; color: #a0aec0;">P&L %</th>
                                            <th style="padding: 8px; text-align: left; color: #a0aec0;">Balance</th>
                                        </tr>
                                    </thead>
                                    <tbody id="{asset['id']}-live-history-tbody">
                                        <tr>
                                            <td colspan="10" style="text-align: center; padding: 20px; color: #6c757d;">No completed trades yet</td>
                                        </tr>
                                    </tbody>
                                </table>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

'''
    new_html.append(asset_html)

# Find the end of assets container
end_found = False
for i in range(asset_start + 1, len(lines)):
    if '</div>' in lines[i] and '<!-- assets-container -->' in lines[i]:
        new_html.append(lines[i])
        end_found = True
        # Add remaining content
        new_html.extend(lines[i+1:])
        break

if not end_found:
    # Find a suitable closing point
    for i in range(len(lines)-1, asset_start, -1):
        if '<!-- Activity Log Sidebar -->' in lines[i]:
            new_html.append('        </div><!-- assets-container -->\n\n')
            new_html.extend(lines[i:])
            break

# Write the new HTML
with open('tradingbot/ui/templates/dashboard.html', 'w', encoding='utf-8') as f:
    f.writelines(new_html)

print("Dashboard HTML completely rebuilt with proper structure!")
print("All asset sections now have:")
print("- Active Positions tables")
print("- Activity Logs")
print("- Trade History tables")
print("\nPlease restart the bot to see the changes.")