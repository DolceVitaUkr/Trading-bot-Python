"""
Fix the HTML structure to properly show the new UI sections
"""

# Read the current dashboard HTML
html_path = 'tradingbot/ui/templates/dashboard.html'
with open(html_path, 'r', encoding='utf-8') as f:
    content = f.read()

# Let's check if we have the malformed HTML sections
if 'forex_options-live-positions-section' in content and content.find('forex_options-live-positions-section') < 200:
    print("Found malformed HTML structure, fixing...")
    
    # Find the position of the first malformed section
    malformed_pos = content.find('<div class="positions-section" id="forex_options-live-positions-section"')
    
    # Find where the metrics-grid section should end (before the malformed section)
    metrics_end = content.rfind('</div>', 0, malformed_pos)
    
    # Remove everything from after metrics-grid to the malformed section
    content = content[:metrics_end + 6] + content[malformed_pos:]
    
    # Now find and fix each asset section to have the proper structure
    assets = ['crypto', 'futures', 'forex', 'forex_options']
    modes = ['paper', 'live']
    
    for asset in assets:
        for mode in modes:
            # Find the trading section for this asset/mode
            section_id = f'{asset}-{mode}-section'
            section_start = content.find(f'id="{section_id}"')
            
            if section_start == -1:
                continue
                
            # Find the end of the metrics-grid for this section
            metrics_search_start = section_start
            metrics_grid_end = content.find('</div><!-- metrics-grid -->', metrics_search_start)
            
            if metrics_grid_end == -1:
                # Try alternative ending pattern
                metrics_grid_end = content.find('</div>\n                    </div>', metrics_search_start)
                if metrics_grid_end != -1:
                    metrics_grid_end += len('</div>\n                    </div>')
            else:
                metrics_grid_end += len('</div><!-- metrics-grid -->')
            
            # Check if we already have the new sections
            if content.find(f'{asset}-{mode}-positions-section', metrics_grid_end, metrics_grid_end + 500) > 0:
                print(f"Sections already exist for {asset} {mode}")
                continue
            
            # Insert the new sections after metrics-grid
            new_sections = f'''
                        <!-- Active Positions -->
                        <div class="positions-section" id="{asset}-{mode}-positions-section" style="margin-top: 10px;">
                            <div class="section-header" style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                                <h5 style="margin: 0; font-size: 14px; color: #e2e8f0;">
                                    <i class="fas fa-chart-line"></i> Active Positions
                                </h5>
                                <div class="positions-controls" style="display: flex; gap: 10px;">
                                    <input type="text" id="{asset}-{mode}-positions-search" placeholder="Search symbol..." style="font-size: 12px; padding: 4px 8px; background: #1a202c; color: #e2e8f0; border: 1px solid #2d3748; border-radius: 4px; width: 120px;">
                                    <select id="{asset}-{mode}-positions-sort" class="positions-sort" style="font-size: 12px; padding: 4px 8px; background: #1a202c; color: #e2e8f0; border: 1px solid #2d3748; border-radius: 4px;">
                                        <option value="symbol">Symbol</option>
                                        <option value="pnl-desc">P&L High to Low</option>
                                        <option value="pnl-asc">P&L Low to High</option>
                                        <option value="size-desc">Size Large to Small</option>
                                    </select>
                                </div>
                            </div>
                            <div class="table-wrapper" style="max-height: 120px; overflow-y: auto; background: #1a202c; border-radius: 6px;">
                                <table class="positions-table" id="{asset}-{mode}-positions-table" style="width: 100%; font-size: 12px;">
                                    <thead style="position: sticky; top: 0; background: #2d3748; z-index: 10;">
                                        <tr>
                                            <th style="padding: 8px; text-align: left; color: #a0aec0; cursor: pointer;" onclick="sortPositions('{asset}', '{mode}', 'symbol')">Symbol ↕</th>
                                            <th style="padding: 8px; text-align: left; color: #a0aec0; cursor: pointer;" onclick="sortPositions('{asset}', '{mode}', 'side')">Side ↕</th>
                                            <th style="padding: 8px; text-align: left; color: #a0aec0; cursor: pointer;" onclick="sortPositions('{asset}', '{mode}', 'size')">Size ↕</th>
                                            <th style="padding: 8px; text-align: left; color: #a0aec0;">Entry</th>
                                            <th style="padding: 8px; text-align: left; color: #a0aec0;">Current</th>
                                            <th style="padding: 8px; text-align: left; color: #a0aec0; cursor: pointer;" onclick="sortPositions('{asset}', '{mode}', 'pnl')">P&L ↕</th>
                                            <th style="padding: 8px; text-align: left; color: #a0aec0;">TP/SL</th>
                                        </tr>
                                    </thead>
                                    <tbody id="{asset}-{mode}-positions-tbody">
                                        <tr>
                                            <td colspan="7" style="text-align: center; padding: 20px; color: #6c757d;">No active positions</td>
                                        </tr>
                                    </tbody>
                                </table>
                            </div>
                        </div>
                        
                        <!-- Activity Log for {asset} {mode} -->
                        <div class="activity-log-section" id="{asset}-{mode}-activity-section" style="margin-top: 15px;">
                            <div class="section-header" style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                                <h5 style="margin: 0; font-size: 14px; color: #e2e8f0;">
                                    <i class="fas fa-history"></i> Activity Log
                                </h5>
                                <div class="activity-controls" style="display: flex; gap: 10px;">
                                    <select id="{asset}-{mode}-activity-filter" class="activity-filter" style="font-size: 12px; padding: 4px 8px; background: #1a202c; color: #e2e8f0; border: 1px solid #2d3748; border-radius: 4px;">
                                        <option value="all">All Activities</option>
                                        <option value="trade">Trades Only</option>
                                        <option value="position">Position Changes</option>
                                        <option value="error">Errors Only</option>
                                    </select>
                                </div>
                            </div>
                            <div class="activity-list" id="{asset}-{mode}-activity-list" style="max-height: 150px; overflow-y: auto; background: #1a202c; border-radius: 6px; padding: 8px;">
                                <div class="no-activity" style="text-align: center; color: #6c757d; padding: 20px; font-size: 12px;">No activity yet</div>
                            </div>
                        </div>
                        
                        <!-- Trade History for {asset} {mode} -->
                        <div class="history-section" id="{asset}-{mode}-history-section" style="margin-top: 15px;">
                            <div class="section-header" style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                                <h5 style="margin: 0; font-size: 14px; color: #e2e8f0;">
                                    <i class="fas fa-table"></i> Trade History
                                </h5>
                                <div class="history-controls" style="display: flex; gap: 10px;">
                                    <input type="text" id="{asset}-{mode}-history-search" placeholder="Search..." style="font-size: 12px; padding: 4px 8px; background: #1a202c; color: #e2e8f0; border: 1px solid #2d3748; border-radius: 4px; width: 120px;">
                                    <select id="{asset}-{mode}-history-sort" class="history-sort" style="font-size: 12px; padding: 4px 8px; background: #1a202c; color: #e2e8f0; border: 1px solid #2d3748; border-radius: 4px;">
                                        <option value="time-desc">Newest First</option>
                                        <option value="time-asc">Oldest First</option>
                                        <option value="pnl-desc">Highest P&L</option>
                                        <option value="pnl-asc">Lowest P&L</option>
                                    </select>
                                </div>
                            </div>
                            <div class="table-wrapper" style="max-height: 200px; overflow-y: auto; background: #1a202c; border-radius: 6px;">
                                <table class="history-table" id="{asset}-{mode}-history-table" style="width: 100%; font-size: 12px;">
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
                                    <tbody id="{asset}-{mode}-history-tbody">
                                        <tr>
                                            <td colspan="10" style="text-align: center; padding: 20px; color: #6c757d;">No completed trades yet</td>
                                        </tr>
                                    </tbody>
                                </table>
                            </div>
                        </div>'''
            
            # Insert at the right position
            if metrics_grid_end > 0:
                content = content[:metrics_grid_end] + new_sections + content[metrics_grid_end:]
                print(f"Added new sections for {asset} {mode}")

# Save the fixed HTML
with open(html_path, 'w', encoding='utf-8') as f:
    f.write(content)

print("HTML structure fixed!")
print("Please restart the bot to see the changes.")