// --- Health banner monitor ---
// Define jget if not already defined
if (typeof jget === 'undefined') {
    async function jget(u){
        const r = await fetch(u);
        if (!r.ok) throw new Error(await r.text());
        return r.json();
    }
}

async function _checkHealth(){
  try{
    const h = await jget('/status');
    let bad = [];
    if (h.bybit && h.bybit.ok === false) bad.push('Bybit');
    if (h.ibkr && h.ibkr.ok === false) bad.push('IBKR');
    let bar = document.getElementById('healthBanner');
    if (bad.length){
      if (!bar){
        bar = document.createElement('div');
        bar.id = 'healthBanner';
        bar.style.position='fixed'; bar.style.top='0'; bar.style.left='0'; bar.style.right='0';
        bar.style.zIndex='9999'; bar.style.padding='8px 12px'; bar.style.background='#7f1d1d'; bar.style.color='#fff';
        bar.style.fontSize='14px'; bar.style.textAlign='center';
        document.body.appendChild(bar);
        // push body down slightly to avoid overlap if you have a fixed header
        document.body.style.paddingTop = '34px';
      }
      bar.textContent = 'Connectivity issues: ' + bad.join(', ') + ' — retrying...';
    }else{
      if (bar){ bar.remove(); document.body.style.paddingTop=''; }
    }
  }catch(e){ /* ignore */ }
}
(function(){
  setInterval(_checkHealth, 10000);
  // initial after boot
  setTimeout(_checkHealth, 2000);
})();