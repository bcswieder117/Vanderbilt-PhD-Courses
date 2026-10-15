"""Reproducible Silverbox Neural ODE demonstration; CPU only, no downloads."""
from pathlib import Path
import argparse,csv,hashlib,json,os,sys,time,faulthandler
for _name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[_name]='1'
os.environ['MPLBACKEND']='Agg'
START=time.perf_counter()
faulthandler.enable()
faulthandler.dump_traceback_later(75,repeat=False)
def progress(message):
    print(f'[{time.perf_counter()-START:6.1f} s] {message}',flush=True)
progress('Importing NumPy, PyTorch, SciPy, and torchdiffeq')
import numpy as np
import torch
from torch import nn
from torchdiffeq import odeint
from scipy.linalg import logm
import scipy,torchdiffeq,matplotlib
progress(f'Loaded PyTorch {torch.__version__}, NumPy {np.__version__}, SciPy {scipy.__version__}')

torch.set_num_threads(1)
torch.set_default_dtype(torch.float64)
torch.use_deterministic_algorithms(True)
ROOT=Path(__file__).resolve().parent
DEFAULT_CONFIG={'seed':17,'updates':120,'batch_size':16,'window_steps':24,
                'learning_rate':0.002,'hidden_width':16,'solver_step_samples':0.5,
                'initialization_samples':50,'validation_steps':256,'training_log_interval':20}

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def load_data(root=ROOT):
    manifest=json.loads((root/'data/source_manifest.json').read_text())
    data={}
    for name,record in manifest['classroom_subsets'].items():
        path=root/'data'/record['file']
        if digest(path)!=record['sha256']:raise ValueError(f'Dataset checksum differs: {path.name}')
        arr=np.genfromtxt(path,delimiter=',',names=True)
        if len(arr)!=record['rows']:raise ValueError('Dataset row count differs')
        if not np.all(np.diff(arr['sample_index'])==1):raise ValueError('Noncontiguous data record')
        data[name]=arr
    return data,manifest

def prepare(data):
    # Only training data define the centering and scales.
    tr=data['train']
    norm={'u_mean':float(tr['input_voltage_V'].mean()),'u_scale':float(tr['input_voltage_V'].std()),
          'y_mean':float(tr['output_voltage_V'].mean()),'y_scale':float(tr['output_voltage_V'].std())}
    normalized={name:(torch.tensor((a['input_voltage_V']-norm['u_mean'])/norm['u_scale']),
                      torch.tensor((a['output_voltage_V']-norm['y_mean'])/norm['y_scale']))for name,a in data.items()}
    u,y=normalized['train'];x=torch.stack((y[1:-1],y[:-2],u[1:-1]),dim=1).numpy()
    coef=np.linalg.lstsq(x,y[2:].numpy(),rcond=None)[0]
    # Continuous warm start for a two-coordinate output-history representation.
    discrete=np.array([[coef[0],coef[1],coef[2]],[1.,0.,0.],[0.,0.,1.]])
    continuous=np.real_if_close(logm(discrete))
    if np.iscomplexobj(continuous):raise ValueError('Linear warm start has a complex logarithm')
    return normalized,norm,torch.tensor(continuous[:2,:2]),torch.tensor(continuous[:2,2:3])

class InputDrivenODE(nn.Module):
    """Two latent coordinates; augmentation optionally adds one zero-initialized coordinate."""
    def __init__(self,A,B,width=16,augmented=False,linear=False):
        super().__init__();self.dimension=3 if augmented else 2;self.linear=linear
        aa=torch.zeros(self.dimension,self.dimension);aa[:2,:2]=A
        bb=torch.zeros(self.dimension,1);bb[:2]=B
        self.register_buffer('A',aa);self.register_buffer('B',bb)
        if not linear:
            self.network=nn.Sequential(nn.Linear(self.dimension+1,width),nn.Tanh(),
                nn.Linear(width,width),nn.Tanh(),nn.Linear(width,self.dimension))
            with torch.no_grad():
                self.network[-1].weight.mul_(0.01);self.network[-1].bias.zero_()
        self.input_path=None
    def set_input(self,values):
        self.input_path=values if values.ndim==3 else values[:,None,None]
    def forward(self,t,state):
        # tau is time in sample intervals. Piecewise-linear input interpolation.
        i=max(0,min(int(t.detach()),len(self.input_path)-2));fraction=t-i
        u=self.input_path[i]+fraction*(self.input_path[i+1]-self.input_path[i])
        derivative=state@self.A.T+u@self.B.T
        if not self.linear:derivative=derivative+self.network(torch.cat((state,u),dim=-1))
        return derivative

def initial_state(model,y,index):
    # Uses only the current and previous output in the permitted prefix.
    state=torch.zeros(1,model.dimension);state[0,:2]=torch.stack((y[index],y[index-1]))
    return state

@torch.no_grad()
def forecast(model,record,config,step=None,limit=None):
    u,y=record;start=config['initialization_samples']-1
    stop=len(y) if limit is None else min(len(y),start+limit+1)
    model.set_input(u[start:stop]);times=torch.arange(stop-start,dtype=torch.float64)
    pred=odeint(model,initial_state(model,y,start),times,method='rk4',
                options={'step_size':step or config['solver_step_samples']})[:,0,0]
    return pred.numpy(),start

def train_model(A,B,records,config,augmented=False):
    torch.manual_seed(config['seed']);generator=torch.Generator().manual_seed(config['seed'])
    model=InputDrivenODE(A,B,config['hidden_width'],augmented=augmented)
    optimizer=torch.optim.Adam(model.parameters(),lr=config['learning_rate'],foreach=False)
    u,y=records['train'];length=config['window_steps'];batch=config['batch_size']
    times=torch.arange(length+1,dtype=torch.float64);history=[];best=None;best_loss=float('inf');best_update=0
    for update in range(1,config['updates']+1):
        starts=torch.randint(1,len(y)-length,(batch,),generator=generator)
        indices=starts[None,:]+torch.arange(length+1)[:,None]
        model.set_input(u[indices][...,None]);state=torch.zeros(batch,model.dimension)
        state[:,:2]=torch.stack((y[starts],y[starts-1]),dim=1)
        optimizer.zero_grad(set_to_none=True)
        prediction=odeint(model,state,times,method='rk4',options={'step_size':config['solver_step_samples']})[...,0]
        loss=(prediction-y[indices]).square().mean()
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite training loss')
        loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),10.0);optimizer.step()
        if update==1 or update%config['training_log_interval']==0 or update==config['updates']:
            estimate,start=forecast(model,records['validation'],config,limit=config['validation_steps'])
            target=records['validation'][1][start:start+len(estimate)].numpy()
            val=float(np.mean((estimate[1:]-target[1:])**2))
            history.append({'update':update,'training_loss':float(loss.detach()),'validation_mse_normalized':val})
            if val<best_loss:
                best_loss=val;best_update=update;best={k:v.detach().clone()for k,v in model.state_dict().items()}
            progress(f'{"Augmented"if augmented else "Neural"} ODE update {update}/{config["updates"]}, validation MSE {val:.5f}')
    model.load_state_dict(best);model.eval()
    return model,history,best_update

def execute(config,output):
    output.mkdir(parents=True,exist_ok=True)
    progress('Reading and validating four Silverbox CSV files')
    data,manifest=load_data();records,norm,A,B=prepare(data)
    progress('Fitting Neural ODE from scratch')
    node,hist,node_update=train_model(A,B,records,config)
    progress('Fitting augmented Neural ODE from scratch')
    anode,ahist,anode_update=train_model(A,B,records,config,augmented=True)
    linear=InputDrivenODE(A,B,linear=True)
    models={'Linear ODE':linear,'Neural ODE':node,'Augmented Neural ODE':anode}
    scores={};predictions={}
    for split in ['test','high_amplitude']:
        progress(f'Forecasting {split} without output feedback')
        predictions[split]={}
        for label,model in models.items():
            estimate,start=forecast(model,records[split],config)
            pred=estimate*norm['y_scale']+norm['y_mean'];true=data[split]['output_voltage_V'][start:]
            err=pred[1:]-true[1:]
            scores.setdefault(label,{})[split]={'rmse_mV':float(1000*np.sqrt(np.mean(err**2))),
                'fit_percent':float(100*(1-np.linalg.norm(err)/np.linalg.norm(true[1:]-np.mean(true[1:]))))}
            predictions[split][label]=pred
        arr=data[split][start:]
        with(output/f'{split}_predictions.csv').open('w',newline='')as f:
            w=csv.writer(f);w.writerow(['sample_index','relative_time_s','input_V','measured_output_V','linear_ode_V','neural_ode_V','augmented_neural_ode_V'])
            for j,row in enumerate(arr):
                w.writerow([int(row['sample_index']),format(j*manifest['sampling_interval_s'],'.17g'),format(row['input_voltage_V'],'.17g'),format(row['output_voltage_V'],'.17g'),*[format(predictions[split][k][j],'.17g')for k in models]])
    progress('Checking RK4 step halving')
    fine,start=forecast(node,records['test'],config,step=config['solver_step_samples']/2)
    step_diff=float(1000*np.max(abs((fine*norm['y_scale']+norm['y_mean'])-predictions['test']['Neural ODE'])))
    model_file=output/'model_weights.json'
    weights={name:{k:v.detach().tolist()for k,v in model.state_dict().items()}for name,model in models.items()}
    model_file.write_text(json.dumps(weights,indent=2)+'\n')
    val_coarse,vi=forecast(node,records['validation'],config,limit=config['validation_steps'])
    val_fine,_=forecast(node,records['validation'],config,step=config['solver_step_samples']/2,limit=config['validation_steps'])
    val_step_diff=float(1000*norm['y_scale']*np.max(abs(val_coarse-val_fine)))
    result={'revision':'R4-Silverbox','config':config,'normalization_training_only':norm,'sampling_frequency_Hz':manifest['sampling_frequency_Hz'],
        'source_archive_sha256':manifest['source_archive_sha256'],'data_sha256':{k:v['sha256']for k,v in manifest['classroom_subsets'].items()},
        'code_sha256':digest(__file__),'weights_sha256':digest(model_file),'scores':scores,'selected_updates':{'Neural ODE':node_update,'Augmented Neural ODE':anode_update},
        'parameter_counts':{k:sum(p.numel()for p in m.parameters())for k,m in models.items()},'history':{'Neural ODE':hist,'Augmented Neural ODE':ahist},
        'step_halving_max_change_mV':step_diff,'validation_step_halving_max_change_mV':val_step_diff,
        'versions':{'python':sys.version.split()[0],'torch':str(torch.__version__),
                    'numpy':np.__version__,'scipy':scipy.__version__,
                    'torchdiffeq':torchdiffeq.__version__,'matplotlib':matplotlib.__version__},
        'scored_samples_per_record':len(data['test'])-config['initialization_samples'],
        'evaluation':'Two classroom subsets; first 50 measured outputs initialize state; all later predictions use input only. RMSE excludes prefix. Not full benchmark results.',
        'elapsed_seconds':time.perf_counter()-START}
    (output/'metrics.json').write_text(json.dumps(result,indent=2)+'\n')
    render_plots(data,predictions,result,output)
    result['elapsed_seconds']=time.perf_counter()-START
    (output/'metrics.json').write_text(json.dumps(result,indent=2)+'\n')
    faulthandler.cancel_dump_traceback_later()
    progress('Complete. Metrics and predictions written.')
    print(json.dumps({'scores':scores,'step_halving_max_change_mV':step_diff,'elapsed_seconds':result['elapsed_seconds']},indent=2))
    return result

def render_plots(data,predictions,result,output):
    progress('Rendering full response, residual, training, and numerical checks')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    colors={'Linear ODE':'#777777','Neural ODE':'#946e24','Augmented Neural ODE':'#a4343a'}
    def save(fig,name):
        fig.tight_layout();fig.savefig(output/name,dpi=150,bbox_inches='tight');plt.close(fig)
    tr=data['train'];t=tr['time_s']-tr['time_s'][0]
    fig,ax=plt.subplots(2,1,figsize=(10,5),sharex=True)
    ax[0].plot(t,tr['input_voltage_V']*1000,color='#777777',lw=.7);ax[0].set_ylabel('Input (mV)')
    ax[1].plot(t,tr['output_voltage_V']*1000,color='#1c1c1c',lw=.7);ax[1].set_ylabel('Output (mV)');ax[1].set_xlabel('Time in Training Record (s)')
    save(fig,'training_data.png')
    start=result['config']['initialization_samples']-1
    diagnostics={}
    for split,file,title in [('test','held_out.png','Full Multisine Test Rollout'),
                             ('high_amplitude','amplitude_test.png','Full Changed-Excitation Rollout')]:
        actual=data[split]['output_voltage_V'][start:]*1000
        prediction=predictions[split]['Neural ODE']*1000
        tt=np.arange(len(actual))/result['sampling_frequency_Hz']
        error=prediction-actual
        full=error[1:];early=error[1:180]
        diagnostics[split]={'plotted_response_points':len(actual),'scored_forecast_points':len(full),
            'duration_s':float(tt[-1]),'first_179_forecast_rmse_mV':float(np.sqrt(np.mean(early**2))),
            'full_rmse_mV':float(np.sqrt(np.mean(full**2))),
            'maximum_absolute_error_mV':float(np.max(abs(full))),
            'residual_definition':'prediction minus measurement',
            'response_axis_mV':[-350,350],'residual_axis_mV':[-200,200]}
        assert max(abs(actual))<350 and max(abs(prediction))<350
        assert max(abs(full))<200
        fig,ax=plt.subplots(2,1,figsize=(11,6.5),sharex=True,
                            gridspec_kw={'height_ratios':[2.2,1.3]})
        ax[0].plot(tt,actual,ls='none',marker='.',ms=2,color='#777777',label='Measured Samples')
        ax[0].plot(tt,prediction,color=colors['Neural ODE'],lw=.8,label='Neural ODE')
        ax[0].set(title=title,ylabel='Output (mV)',ylim=(-350,350),xlim=(0,3.3))
        ax[0].legend(ncol=2,loc='upper right')
        ax[1].axhline(0,color='#777777',lw=.7)
        ax[1].plot(tt[1:],full,color='#a4343a',lw=.8)
        ax[1].set(title='Residual: Prediction Minus Measurement',ylabel='Error (mV)',
                  xlabel='Time after Initialization (s)',ylim=(-200,200),xlim=(0,3.3))
        ax[1].set_yticks([-200,-100,0,100,200])
        ax[1].text(.01,.95,f"Full RMSE: {diagnostics[split]['full_rmse_mV']:.3f} mV; "
                    f"maximum |error|: {diagnostics[split]['maximum_absolute_error_mV']:.3f} mV",
                    transform=ax[1].transAxes,va='top',fontsize=10)
        save(fig,file)
    (output/'plot_diagnostics.json').write_text(json.dumps(diagnostics,indent=2)+'\n')
    fig,ax=plt.subplots(figsize=(9,4))
    for label,h in result['history'].items():
        ax.semilogy([x['update']for x in h],[x['validation_mse_normalized']for x in h],
                    label=label,color=colors[label],marker='o',ms=4)
    ax.set(xlabel='Adam Update',ylabel='Validation MSE (Normalized)',
           title='Checkpoint Selection Uses Validation Only');ax.legend();save(fig,'training_history.png')
    fig,ax=plt.subplots(figsize=(9,4))
    names=list(result['scores']);x=np.arange(len(names));w=.36
    for j,(split,title,color) in enumerate([('test','Multisine Subset','#946e24'),
                   ('high_amplitude','Changed-Excitation Subset','#a4343a')]):
        bars=ax.bar(x+(j-.5)*w,[result['scores'][name][split]['rmse_mV']for name in names],w,label=title,color=color)
        ax.bar_label(bars,fmt='%.2f',padding=3,fontsize=9)
    ax.set_xticks(x,names);ax.set_ylabel('Output RMSE (mV)');ax.set_ylim(0,33);ax.legend();save(fig,'augmentation.png')
    fig,ax=plt.subplots(figsize=(9,3.5))
    ax.bar(['Validation Step Check','Test Step Check'],[result['validation_step_halving_max_change_mV'],
            result['step_halving_max_change_mV']],color=['#777777','#946e24'])
    ax.set(ylabel='Maximum Output Change (mV)',title='RK4 Step Halving with the Same Fitted Weights');save(fig,'step_check.png')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=ROOT/'results');parser.add_argument('--config',type=Path)
    args=parser.parse_args();cfg=DEFAULT_CONFIG.copy()
    if args.config:cfg.update(json.loads(args.config.read_text()))
    execute(cfg,args.output)
