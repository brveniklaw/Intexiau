// Rebuilds index.html as real HTML from the design source assets/home/app.dc.html.
// Edit that file (or scripts/home-head.html), then run: node scripts/prerender.mjs
// The page shows the static HTML first; JavaScript then upgrades it to the animated version.
import { chromium } from 'playwright';
import http from 'node:http';
import { readFileSync, writeFileSync, existsSync, statSync } from 'node:fs';
import { join, extname, resolve } from 'node:path';
const root = resolve(new URL('..', import.meta.url).pathname);
const RUNTIME = '/assets/home/e3a84ca6-7c39-41be-b3a0-c80d74f70052.js';
const reactDir = process.env.REACT_MODULES; // folder holding react and react-dom from npm
const head = readFileSync(join(root, 'scripts/home-head.html'), 'utf8');
const app = readFileSync(join(root, 'assets/home/app.dc.html'), 'utf8');
const types = { '.js': 'text/javascript', '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg', '.png': 'image/png', '.woff2': 'font/woff2', '.html': 'text/html' };
const server = http.createServer((q, r) => {
  const p = decodeURIComponent(q.url.split('?')[0]);
  if (p === '/') { r.setHeader('content-type', 'text/html'); return r.end(`${head}<script src="${RUNTIME}"></script></head><body>${app}</body></html>`); }
  const f = join(root, p);
  if (existsSync(f) && statSync(f).isFile()) { r.setHeader('content-type', types[extname(f)] || 'application/octet-stream'); return r.end(readFileSync(f)); }
  r.statusCode = 404; r.end();
}).listen(0);
const url = `http://localhost:${server.address().port}/`;
const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 1280, height: 900 } });
await page.route(/unpkg\.com\/(react|react-dom)@/, route => {
  const m = route.request().url().match(/unpkg\.com\/(react|react-dom)@[^/]+\/(.*)/);
  route.fulfill({ contentType: 'text/javascript', headers: { 'access-control-allow-origin': '*' }, body: readFileSync(join(reactDir, m[1], m[2])) });
});
await page.route(/images\.unsplash\.com|fonts\.g/, r => r.abort());
const errors = []; page.on('pageerror', e => errors.push(e.message));
await page.goto(url); await page.waitForSelector('#dc-root h1'); await page.waitForTimeout(800);
const out = await page.evaluate(() => {
  const c = document.getElementById('dc-root').cloneNode(true);
  c.querySelectorAll('[data-reveal]').forEach(e => { e.style.opacity = ''; e.style.transform = ''; e.style.transition = ''; e.removeAttribute('data-rev'); });
  c.removeAttribute('id'); c.id = 'prerendered';
  const css = [...document.head.querySelectorAll('style')].map(s => s.textContent).join('\n');
  return { html: c.outerHTML, css };
});
await browser.close(); server.close();
if (errors.length) throw new Error(errors.join('\n'));
if (out.html.includes('{{')) throw new Error('template braces left in output');
const boot = `<script>
(function(){var pre=document.getElementById('prerendered');
fetch('/assets/home/app.dc.html').then(function(r){if(!r.ok)throw 0;return r.text()}).then(function(t){
var h=document.createElement('div');h.innerHTML=t;while(h.firstChild)document.body.appendChild(h.firstChild);
new MutationObserver(function(_,o){var d=document.getElementById('dc-root');if(d&&d.firstChild){o.disconnect();pre.remove()}}).observe(document.body,{childList:true,subtree:true});
var s=document.createElement('script');s.src='${RUNTIME}';document.body.appendChild(s)}).catch(function(){})})();
</script>`;
writeFileSync(join(root, 'index.html'), `${head}<style data-prerender>x-dc{display:none}${out.css}</style>\n</head>\n<body>\n${out.html}\n${boot}\n</body></html>\n`);
console.log('index.html rebuilt', out.html.length);
