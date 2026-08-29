import { chromium } from 'playwright';
import { mkdirSync } from 'fs';
const out='/tmp/shots2'; mkdirSync(out,{recursive:true});
const b = await chromium.launch({ executablePath: '/opt/pw-browsers/chromium' });
const p = await b.newPage({ viewport:{width:1440,height:1100}, deviceScaleFactor:2 });
const errors=[]; p.on('console',m=>{if(m.type()==='error')errors.push(m.text())}); p.on('pageerror',e=>errors.push(String(e)));

await p.goto('http://localhost:3000',{waitUntil:'networkidle'});
await p.getByLabel(/Safe repair/i).check().catch(()=>p.locator('input[value="safe_repair"]').check());
await p.getByRole('button',{name:/Start verification/i}).click();
await p.waitForURL(/\/claims\//,{timeout:20000});
await p.waitForTimeout(3000);
await p.getByRole('button',{name:/Experiment plan/}).click();
await p.waitForTimeout(500);
await p.getByRole('button',{name:/^Approve/}).click();

// wait until the claim reports it is blocked on a repair approval
let blocked=false;
for(let i=0;i<40;i++){
  const r=await p.request.get('http://localhost:3000/api/backend/claims');
  const c=(await r.json())[0];
  if(c.state==='awaiting_approval' && /repair/i.test(c.latest_action||'')){blocked=true;break;}
  if(c.state==='verdict')break;
  await p.waitForTimeout(1500);
}
console.log('reached repair-approval gate:', blocked);
await p.getByRole('button',{name:/^Queue/}).click();
await p.waitForTimeout(1200);
await p.screenshot({path:`${out}/01-queue-repair-approval.png`,fullPage:true});

// approve the held repair from the queue
const btn=p.getByRole('button',{name:/Approve this run/});
const n=await btn.count();
console.log('per-job approve buttons:', n);
if(n>0){ await btn.first().click(); }

// Under safe repair, later rounds need plan approval too.
let claimId=null;
for(let i=0;i<80;i++){
  const r=await p.request.get('http://localhost:3000/api/backend/claims');
  const c=(await r.json())[0]; claimId=c.id;
  if(c.state==='verdict')break;
  if(c.state==='awaiting_approval'){
    const s=await (await p.request.get(`http://localhost:3000/api/backend/claims/${c.id}`)).json();
    const plan=s.plans.find(x=>x.status==='awaiting_approval');
    if(plan){
      await p.request.post(`http://localhost:3000/api/backend/claims/${c.id}/plans/${plan.id}/decision`,
        {data:{approved:true,decided_by:'e2e'}});
      console.log('approved round', plan.round_index+1);
    }
  }
  await p.waitForTimeout(1500);
}
await p.waitForTimeout(2500);
await p.getByRole('button',{name:/Final report/}).click();
await p.waitForTimeout(1000);
await p.screenshot({path:`${out}/02-report-after-repair.png`,fullPage:true});
const r=await p.request.get('http://localhost:3000/api/backend/claims');
console.log('final state:', (await r.json())[0].state);
console.log('console errors:', errors.length? errors : 'none');
await b.close();
