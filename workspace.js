'use strict';
const el = id => document.getElementById(id);
const show = (id, visible) => { el(id).hidden = !visible; };
let file, previewURL, generation = 0, ready = false, busy = false, session;
const CLASS_NAMES = ['Glioma','Meningioma','No Tumor','Pituitary'];
const ortPromise = import('./ort.wasm.bundle.min.mjs');
function message(text = '', error = false) {
  el('feedback').textContent = text;
  el('feedback').classList.toggle('error', error);
}
function controls() {
  el('run').disabled = !file || !ready || busy;
}
function reset() {
  generation++;
  busy = false;
  file = null;
  if (previewURL) URL.revokeObjectURL(previewURL);
  previewURL = null;
  el('preview').removeAttribute('src');
  el('file').value = '';
  el('filename').textContent = '';
  el('browse').textContent = 'Browse files';
  el('bars').replaceChildren();
  el('winner').textContent = '';
  el('workspace').classList.remove('analyzing');
  ['preview','preview-frame','remove','output','loading'].forEach(id => show(id, false));
  ['empty','waiting'].forEach(id => show(id, true));
  message();
  controls();
}
async function select(candidate) {
  reset();
  const id = generation;
  if (!candidate) return;
  if (!['image/png','image/jpeg'].includes(candidate.type)) return message('Choose a PNG or JPEG image.', true);
  if (candidate.size > 10 * 1024 * 1024) return message('Choose a file of 10 MiB or less.', true);
  previewURL = URL.createObjectURL(candidate);
  const img = el('preview');
  img.src = previewURL;
  try {
    await img.decode();
    if (id !== generation) return;
    if (img.naturalWidth * img.naturalHeight > 16000000) throw Error('Use an image with at most 16 million pixels.');
    file = candidate;
    el('filename').textContent = file.name;
    el('browse').textContent = 'Replace image';
    show('empty', false);
    show('preview-frame', true);
    show('preview', true);
    show('remove', true);
    controls();
  } catch (err) {
    if (id !== generation) return;
    reset();
    message(err.name === 'EncodingError' ? 'This image could not be decoded. Choose another PNG or JPEG.' : err.message, true);
  }
}
el('browse').onclick = () => el('file').click();
el('file').onchange = () => select(el('file').files[0]);
for (const name of ['dragover','dragleave','drop']) {
  el('drop').addEventListener(name, event => {
    event.preventDefault();
    el('drop').classList.toggle('drag', name === 'dragover');
    if (name === 'drop') select(event.dataTransfer.files[0]);
  });
}
el('remove').onclick = el('another').onclick = () => {reset(); el('browse').focus();};
async function initializeModel() {
  el('health').textContent = 'Loading browser model';
  controls();
  try {
    const ort = await ortPromise;
    ort.env.wasm.numThreads = 1;
    ort.env.wasm.proxy = false;
    session = await ort.InferenceSession.create(new URL('./model.onnx', import.meta.url).href, {executionProviders:['wasm']});
    ready = true;
    el('health').textContent = 'Model ready · runs locally';
  } catch (error) {
    ready = false;
    el('health').textContent = 'Model unavailable';
    message('The browser model could not be loaded. Refresh the page to retry.', true);
    console.error('NeuroScope browser model initialization failed:', error);
  }
  controls();
}
initializeModel();
async function predictLocally(candidate) {
  const ort = await ortPromise;
  const bitmap = await createImageBitmap(candidate);
  const canvas = document.createElement('canvas');
  canvas.width = 224; canvas.height = 224;
  const context = canvas.getContext('2d', {willReadFrequently:true});
  context.imageSmoothingEnabled = false;
  context.drawImage(bitmap, 0, 0, 224, 224);
  bitmap.close();
  const rgba = context.getImageData(0, 0, 224, 224).data;
  const input = new Float32Array(224 * 224 * 3);
  for (let src = 0, dst = 0; src < rgba.length; src += 4) {
    input[dst++] = rgba[src] / 255;
    input[dst++] = rgba[src + 1] / 255;
    input[dst++] = rgba[src + 2] / 255;
  }
  const result = await session.run({[session.inputNames[0]]:new ort.Tensor('float32', input, [1,224,224,3])});
  const output = Array.from(result[session.outputNames[0]].data, Number);
  if (output.length !== 4 || output.some(value => !Number.isFinite(value) || value < 0 || value > 1)
      || Math.abs(output.reduce((sum,value)=>sum+value,0)-1) > 1e-5) throw Error('The model returned an invalid probability vector.');
  const scores = Object.fromEntries(CLASS_NAMES.map((name,index)=>[name,output[index]]));
  return {classification:CLASS_NAMES[output.indexOf(Math.max(...output))], scores};
}
function render(data) {
  el('winner').textContent = data.classification;
  el('bars').replaceChildren();
  for (const [label, score] of Object.entries(data.scores)) {
    const row = document.createElement('div');
    row.className = 'meter';
    const caption = document.createElement('label');
    caption.textContent = label;
    const percentage = document.createElement('span');
    percentage.textContent = '0.00%';
    caption.append(percentage);
    const bar = document.createElement('progress');
    bar.setAttribute('aria-label', label);
    bar.max = 1;
    bar.value = 0;
    row.style.setProperty('--n', el('bars').children.length);
    row.append(caption,bar);
    el('bars').append(row);
    const reduced = matchMedia('(prefers-reduced-motion: reduce)').matches;
    if (reduced) {bar.value = score; percentage.textContent = (score * 100).toFixed(2) + '%'; continue;}
    const start = performance.now() + (el('bars').children.length - 1) * 90;
    function count(now) {
      if (!row.isConnected) return;
      const amount = Math.max(0, Math.min(1, (now - start) / 800));
      const eased = 1 - Math.pow(1 - amount, 3);
      bar.value = score * eased;
      percentage.textContent = (score * eased * 100).toFixed(2) + '%';
      if (amount < 1) requestAnimationFrame(count);
      else {bar.value = score; percentage.textContent = (score * 100).toFixed(2) + '%';}
    }
    requestAnimationFrame(count);
  }
}
el('run').onclick = async () => {
  if (!file || busy || !ready) return;
  const id = generation;
  busy = true;
  el('workspace').classList.add('analyzing');
  controls();
  message();
  show('waiting', false); show('output', false); show('loading', true);
  try {
    const data = await predictLocally(file);
    if (id !== generation) return;
    render(data);
    show('output', true);
    message('Analysis complete.');
  } catch (err) {
    if (id !== generation) return;
    show('waiting', true);
    message(err.name === 'EncodingError' ? 'This image could not be decoded. Choose another PNG or JPEG.' : err.message, true);
  } finally {
    if (id === generation) {busy = false; el('workspace').classList.remove('analyzing'); show('loading', false); controls();}
  }
};
function menu(open) {
  document.querySelector('nav').classList.toggle('open',open);
  el('menu').setAttribute('aria-expanded',String(open));
  el('menu').setAttribute('aria-label',open?'Close navigation':'Open navigation');
}
el('menu').onclick = () => menu(el('menu').getAttribute('aria-expanded') !== 'true');
document.addEventListener('keydown',e=>{if(e.key==='Escape')menu(false);});
const links = [...el('links').querySelectorAll('a')];
links.forEach((link,i)=>{link.style.setProperty('--n',i);link.onclick=()=>menu(false);});
const observer = new IntersectionObserver(entries=>{
  for (const entry of entries) if(entry.isIntersecting) {
    links.forEach(link=>{
      if(link.hash === '#'+entry.target.id) link.setAttribute('aria-current','location');
      else link.removeAttribute('aria-current');
    });
  }
},{rootMargin:'0px 0px -65% 0px'});
links.forEach(link=>observer.observe(document.querySelector(link.hash)));
const reveal = new IntersectionObserver(entries=>entries.forEach(e=>{
  if(e.isIntersecting){e.target.classList.add('reveal');reveal.unobserve(e.target);}
}));
document.querySelectorAll('.body section').forEach(section=>reveal.observe(section));
const tagline=document.querySelector('.tagline');
tagline.replaceChildren();
const words=['One','scan.','Four','class','probabilities.'].map((word,i)=>{
  if(i===2)tagline.append(document.createElement('br'));
  const span=document.createElement('span');
  span.textContent=word+' ';
  tagline.append(span);
  return span;
});
if (!matchMedia('(prefers-reduced-motion: reduce)').matches) {
  let pending=false;
  const paint=()=>{const amount=(innerHeight*.88-tagline.getBoundingClientRect().top)/120;
    words.forEach((w,i)=>w.style.opacity=amount>i/words.length?'1':'.35');pending=false;};
  addEventListener('scroll',()=>{if(!pending){pending=true;requestAnimationFrame(paint);}},{passive:true});
  paint();
}
document.querySelectorAll('[data-legal]').forEach(btn=>btn.onclick=()=>{
  const privacy=btn.dataset.legal==='privacy';
  el('legal').querySelector('h3').textContent=btn.textContent;
  el('legal').querySelector('p').textContent=privacy?'This application runs the original classifier in your browser. Your selected image is processed locally and is not uploaded. Use non-sensitive sample images.':'This portfolio application is for education and research. Do not use its outputs for diagnosis or treatment decisions.';
  el('legal').showModal();
});
el('close').onclick=()=>el('legal').close();

const artwork = el('hero-art');
const reducedMotion = matchMedia('(prefers-reduced-motion: reduce)');
if (!reducedMotion.matches && matchMedia('(pointer: fine)').matches) {
  artwork.addEventListener('pointermove', event => {
    const box = artwork.getBoundingClientRect();
    const x = (event.clientX - box.left) / box.width - .5;
    const y = (event.clientY - box.top) / box.height - .5;
    artwork.querySelector('.art-image').style.transform = `translate3d(${x * 16}px, ${y * 16}px, 0)`;
  }, {passive:true});
  artwork.addEventListener('pointerleave', () => artwork.querySelector('.art-image').style.transform = '');
  for (const button of document.querySelectorAll('.btn')) {
    button.addEventListener('pointermove', event => {
      const box = button.getBoundingClientRect();
      button.style.transform = `translate3d(${(event.clientX - box.left - box.width / 2) * .08}px, ${(event.clientY - box.top - box.height / 2) * .08}px, 0)`;
    }, {passive:true});
    button.addEventListener('pointerleave', () => button.style.transform = '');
  }
}
const backTop = el('back-top');
backTop.onclick = () => scrollTo({top:0,behavior:reducedMotion.matches?'instant':'smooth'});
let motionScheduled = false;
function updateScrollMotion() {
  const max = document.documentElement.scrollHeight - innerHeight;
  el('scroll-track').firstElementChild.style.transform = `scaleY(${max > 0 ? scrollY / max : 0})`;
  backTop.classList.toggle('visible', scrollY > innerHeight);
  if (!reducedMotion.matches) {
    const travel = Math.max(-24, Math.min(24, scrollY * .045));
    artwork.querySelector('.art-grid').style.transform = `translateY(${travel}px)`;
  }
  motionScheduled = false;
}
addEventListener('scroll', () => {
  if (!motionScheduled) {motionScheduled = true; requestAnimationFrame(updateScrollMotion);}
}, {passive:true});
updateScrollMotion();
const stepObserver = new IntersectionObserver(entries => {
  for (const entry of entries) if (entry.isIntersecting) {
    [...document.querySelectorAll('.steps li')].forEach((step,index) => {
      if (reducedMotion.matches) step.classList.add('drawn');
      else setTimeout(() => step.classList.add('drawn'), index * 140);
    });
    stepObserver.disconnect();
  }
});
stepObserver.observe(el('method'));
if (!reducedMotion.matches) document.body.classList.add('motion-ready');
const detailObserver = new IntersectionObserver(entries => {
  for (const entry of entries) if (entry.isIntersecting) {
    entry.target.classList.add('drawn');
    detailObserver.unobserve(entry.target);
  }
}, {rootMargin:'0px 0px -40px 0px'});
document.querySelectorAll('#faq details').forEach(detail => detailObserver.observe(detail));
