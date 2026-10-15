"""Independently check measurements, forecasts, full charts, residuals, and notes."""
from pathlib import Path
from zipfile import ZipFile
import xml.etree.ElementTree as E
import csv,hashlib,json,math,posixpath
R=Path(__file__).resolve().parent
ns={'a':'http://schemas.openxmlformats.org/drawingml/2006/main','p':'http://schemas.openxmlformats.org/presentationml/2006/main','c':'http://schemas.openxmlformats.org/drawingml/2006/chart','r':'http://schemas.openxmlformats.org/officeDocument/2006/relationships'}
m=json.loads((R/'results/metrics.json').read_text());manifest=json.loads((R/'data/source_manifest.json').read_text());checks=[]
assert m['config']==json.loads((R/'config.json').read_text())
assert m['code_sha256']==hashlib.sha256((R/'silverbox_demo.py').read_bytes()).hexdigest()
for split,s in manifest['classroom_subsets'].items():assert hashlib.sha256((R/'data'/s['file']).read_bytes()).hexdigest()==s['sha256']==m['data_sha256'][split]
records={};independent={}
for split in ['test','high_amplitude']:
 with(R/'results'/f'{split}_predictions.csv').open(newline='')as f:rows=list(csv.DictReader(f))
 with(R/'data'/manifest['classroom_subsets'][split]['file']).open(newline='')as f:raw=list(csv.DictReader(f))[49:]
 assert len(rows)==len(raw)==1999
 for j,(a,b)in enumerate(zip(rows,raw)):
  assert int(a['sample_index'])==int(b['sample_index'])
  assert float(a['measured_output_V'])==float(b['output_voltage_V'])
  assert float(a['input_V'])==float(b['input_voltage_V'])
  assert math.isclose(float(a['relative_time_s']),j/manifest['sampling_frequency_Hz'],abs_tol=1e-14)
 independent[split]={}
 for label,col in [('Linear ODE','linear_ode_V'),('Neural ODE','neural_ode_V'),('Augmented Neural ODE','augmented_neural_ode_V')]:
  errors=[1000*(float(x[col])-float(x['measured_output_V']))for x in rows[1:]]
  rmse=math.sqrt(sum(e*e for e in errors)/len(errors));assert math.isclose(rmse,m['scores'][label][split]['rmse_mV'],rel_tol=1e-13)
  independent[split][label]={'rmse_mV':rmse,'max_abs_mV':max(map(abs,errors))}
 records[split]={'t':[float(x['relative_time_s'])for x in rows],
  'measured':[1000*float(x['measured_output_V'])for x in rows],
  'node':[1000*float(x['neural_ode_V'])for x in rows]}
 records[split]['error']=[a-b for a,b in zip(records[split]['node'],records[split]['measured'])]
checks.append('All six RMSE values recomputed from full forecasts; source measurements and times match local CSVs.')
def close(a,b):
 assert len(a)==len(b),(len(a),len(b))
 assert all(math.isclose(x,y,rel_tol=1e-10,abs_tol=1e-10)for x,y in zip(a,b))
with ZipFile(R/'Neural_ODE_Class_Slides.pptx')as z:
 def text(i):return ' '.join(E.fromstring(z.read(f'ppt/slides/slide{i}.xml')).itertext())
 def charts(i):
  owner=f'ppt/slides/slide{i}.xml';root=E.fromstring(z.read(owner));rels=E.fromstring(z.read(f'ppt/slides/_rels/slide{i}.xml.rels'))
  values=[]
  for c in root.findall('.//c:chart',ns):
   rid=c.get('{'+ns['r']+'}id');target=next(e.get('Target')for e in rels if e.get('Id')==rid)
   values.append(E.fromstring(z.read(posixpath.normpath(posixpath.join('ppt/slides',target)).lstrip('/'))))
  return values
 def values(ser,axis):
  a=ser.findall(f'c:{axis}Val/c:numRef/c:numCache/c:pt/c:v',ns)
  if not a:a=ser.findall(f'c:{axis}Val/c:numLit/c:pt/c:v',ns)
  return [float(e.text)for e in a]
 expected_titles={2:'What Is a Neural ODE?',12:'Paper 1: Five Takeaways',17:'Paper 2: Five Takeaways',24:'Demo Results: Full Test Response and Error',26:'Demo Results: Changed Excitation and Error',36:'Sources: Data, Software, and Evaluation'}
 for i,title in expected_titles.items():assert title in text(i),(i,title)
 for i,word in [(6,'Neural Ordinary'),(13,'Augmented Neural ODEs'),(18,'Silverbox'),(33,'Conclusion'),(34,'Questions')]:assert word in text(i)
 for slide,values0 in {10:['0.41%','0.47%','0.42%'],11:['0.3937','0.1642','0.3202','0.1502','0.1813','0.1346'],16:['96.4%','98.2%','53.7%','60.6%','81.0%','83.5%']}.items():
  for v in values0:assert v in text(slide),(slide,v)
 for label,scores in m['scores'].items():
  assert label in text(25)
  for split in ['test','high_amplitude']:assert f"{scores[split]['rmse_mV']:.3f}"in text(25)
 for i,split in [(24,'test'),(26,'high_amplitude')]:
  cc=charts(i);assert len(cc)==2,(i,len(cc));rec=records[split]
  top=cc[0].findall('.//c:scatterChart/c:ser',ns);bottom=cc[1].findall('.//c:scatterChart/c:ser',ns)
  assert len(top)==2 and len(bottom)==1
  for ser,key in zip(top,['measured','node']):close(values(ser,'x'),rec['t']);close(values(ser,'y'),rec[key])
  close(values(bottom[0],'x'),rec['t'][1:]);close(values(bottom[0],'y'),rec['error'][1:])
  assert len(values(top[0],'y'))==1999 and len(values(bottom[0],'y'))==1998
  for c,minimum,maximum in [(cc[0],-350,350),(cc[1],-200,200)]:
   axes=c.findall('.//c:valAx',ns);scales=[(float(a.find('c:scaling/c:min',ns).get('val')),float(a.find('c:scaling/c:max',ns).get('val')))for a in axes]
   assert (minimum,maximum)in scales,(i,scales)
   for ser in c.findall('.//c:scatterChart/c:ser',ns):assert ser.find('c:smooth',ns).get('val')=='0'
  full=independent[split]['Neural ODE'];assert f"{full['rmse_mV']:.3f}"in text(i);assert f"{full['max_abs_mV']:.3f}"in text(i)
  checks.append(f'Slide {i}: all 1,999 response points and 1,998 signed residuals match forecast CSVs; shared axes and no smoothing.')
 c=charts(27);assert len(c)==1
 for ser,label in zip(c[0].findall('.//c:scatterChart/c:ser',ns),['Neural ODE','Augmented Neural ODE']):
  close(values(ser,'x'),[x['update']for x in m['history'][label]])
  close(values(ser,'y'),[x['validation_mse_normalized']for x in m['history'][label]])
 for i in range(1,37):
  note=' '.join(E.fromstring(z.read(f'ppt/notesSlides/notesSlide{i}.xml')).itertext())
  assert len(note.split('Sources')[0].split())>=40,i
  if i in [24,26]:
   assert '1,998'in note and 'prediction minus measurement'in note and 'mathworks.com'in note and '2103.14516'in note
 alltext=' '.join(text(i)for i in range(1,37))
 assert 'Detail shows the first'not in alltext
 assert 'Larger-amplitude excitation produces substantially larger errors'not in alltext
 checks.append('All 36 slides retain explanatory speaker notes; evaluation references and confounding limits accompany the revised figures.')
nb=json.loads((R/'Neural_ODE_Demo.ipynb').read_text());code=[c for c in nb['cells']if c['cell_type']=='code']
assert len(nb['cells'])==24 and len(code)==13
assert [c['execution_count']for c in code]==list(range(1,14))
assert not any(o.get('output_type')=='error'for c in code for o in c['outputs'])
assert sum('image/png'in o.get('data',{})for c in code for o in c['outputs'])==6
assert not any('base64'in ''.join(c['source'])or'_REFERENCE_DATA'in ''.join(c['source'])for c in code)
report={'status':'passed','slide_count':36,'native_charts':5,'speaker_notes_slides':36,'checks':checks,
 'independent_full_rollout_errors':independent,'code_sha256':m['code_sha256'],
 'powerpoint_sha256':hashlib.sha256((R/'Neural_ODE_Class_Slides.pptx').read_bytes()).hexdigest(),
 'notebook_executed_cells':13,'notebook_figures':6,
 'scope':'Measured classroom subsets with local initialization. Checks verify artifact/data alignment; they do not establish full-benchmark performance or native PowerPoint application behavior.'}
(R/'alignment_report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
