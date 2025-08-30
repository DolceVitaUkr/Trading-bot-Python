"""
Add missing JavaScript functions for activity log and trade history
"""

# Read the current dashboard.js
with open('tradingbot/ui/static/dashboard.js', 'r', encoding='utf-8') as f:
    js_content = f.read()

# Check if functions already exist
if 'updateActivityLog' not in js_content:
    print("Adding updateActivityLog function...")
    
    # Find where to insert (after updatePositionsTable)
    insert_pos = js_content.find('updatePositionsTable(asset, mode, positions) {')
    if insert_pos == -1:
        print("Could not find updatePositionsTable function")
        exit(1)
    
    # Find the end of updatePositionsTable function
    brace_count = 0
    started = False
    end_pos = insert_pos
    for i in range(insert_pos, len(js_content)):
        if js_content[i] == '{':
            brace_count += 1
            started = True
        elif js_content[i] == '}':
            brace_count -= 1
            if started and brace_count == 0:
                end_pos = i + 1
                break
    
    # Insert new functions after updatePositionsTable
    new_functions = '''

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
'''
    
    # Insert the new functions
    js_content = js_content[:end_pos] + new_functions + js_content[end_pos:]
    
    # Write back
    with open('tradingbot/ui/static/dashboard.js', 'w', encoding='utf-8') as f:
        f.write(js_content)
    
    print("Added updateActivityLog and updateTradeHistory functions")
else:
    print("Functions already exist")

# Now add calls to these functions in the update loops
print("\nChecking if functions are being called...")

# Check if we need to add calls in updateAssetStatus
if 'updateActivityLog' not in js_content[js_content.find('updateAssetStatus'):]:
    print("Need to add function calls in update loops...")
    
    # Find the updateAllAssets function
    update_fn_pos = js_content.find('async updateAllAssets() {')
    if update_fn_pos > 0:
        # Find where positions are updated
        positions_update = js_content.find('// Update positions display', update_fn_pos)
        if positions_update > 0:
            # Find the end of that section
            next_section = js_content.find('// Update strategies', positions_update)
            if next_section > 0:
                # Insert activity and history updates
                insert_code = '''
                    // Update activity log
                    if (data.activities) {
                        this.updateActivityLog(asset, mode, data.activities);
                    }
                    
                    // Update trade history
                    if (data.trades) {
                        this.updateTradeHistory(asset, mode, data.trades);
                    }
                    '''
                
                js_content = js_content[:next_section] + insert_code + '\n' + js_content[next_section:]
                
                with open('tradingbot/ui/static/dashboard.js', 'w', encoding='utf-8') as f:
                    f.write(js_content)
                    
                print("Added function calls to update loop")

print("\nJavaScript updates complete!")
print("The UI should now show:")
print("- Active Positions tables")
print("- Activity Logs") 
print("- Trade History tables")
print("\nPlease refresh your browser to see the changes.")