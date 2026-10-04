import fs from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath,pathToFileURL} from 'node:url';
import {Presentation,PresentationFile} from '@oai/artifact-tool';

const repoRoot=path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const buildDir=path.resolve(process.env.TMP_DIR ?? path.join(repoRoot, 'work/overview'));
await fs.mkdir(buildDir,{recursive:true});
const rows=(await fs.readFile(path.join(repoRoot,'outputs/tables/summary.csv'),'utf8')).trim().split('\n');
const keys=rows.shift().split(',');
const summaries=Object.fromEntries(rows.map(line=>{
  const obj=Object.fromEntries(line.split(',').map((v,i)=>[keys[i],v]));return [obj.setting,obj];
}));
const settings=['centralized','iid','mild_non_iid','strong_non_iid','strong_e3'];
const labels=['Centralized','FedAvg IID','FedAvg mild','FedAvg strong','Strong E3R10'];
const metrics=JSON.parse(await fs.readFile(path.join(repoRoot,'outputs/runs/strong_non_iid/seed_42/final_metrics.json'),'utf8')).final_test;
const strongRuns=await Promise.all([42,43,44].map(async seed=>JSON.parse(await fs.readFile(path.join(repoRoot,`outputs/runs/strong_non_iid/seed_${seed}/final_metrics.json`),'utf8')).final_test));
const shirtRecall=strongRuns.reduce((sum,m)=>sum+m.per_class[6].recall,0)/3;
const gap=(100*(Number(summaries.strong_non_iid.accuracy_mean)-Number(summaries.iid.accuracy_mean))).toFixed(2);
const delta=(100*(Number(summaries.strong_e3.accuracy_mean)-Number(summaries.strong_non_iid.accuracy_mean))).toFixed(2);
const p=Presentation.create({slideSize:{width:1280,height:720}});
function text(slide,x,y,w,h,content,size=30,bold=false){
  const shape=slide.shapes.add({geometry:'textbox',position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});
  shape.text=content;
  shape.text.style={typeface:'Arial',fontSize:size,bold,color:'#192e55',autoFit:'none'};
  return shape;
}
const titles=['Problem & Research Question','Method & Experiments','Key Results','Conclusion / Demo'];
const slides=titles.map(title=>{
 const slide=p.slides.add();slide.background.fill='#ffffff';text(slide,64,35,1150,85,title,44,true);text(slide,64,678,1140,28,'Group 9 - Topic 27',16);return slide;
});
text(slides[0],72,150,1140,125,'Federated Image Classification\nunder Non-IID Data',48,true);
text(slides[0],72,315,1140,82,'How does client label skew change global classification quality?',32);
text(slides[0],72,420,1140,80,'10 clients, equal image counts, different label proportions.',32);
text(slides[0],72,520,1140,90,'Compare centralized training, skew severity and local epochs.',32);
text(slides[1],72,145,1140,80,'Fashion-MNIST: 54k train / 6k validation / 10k test',31);
text(slides[1],72,220,1140,85,'Small CNN, 105,866 parameters. Weighted FedAvg in PyTorch.',31);
text(slides[1],72,320,1140,70,'Setup 1: Centralized versus FedAvg IID',31);
text(slides[1],72,400,1140,70,'Setup 2: IID, mild and strong label skew',31);
text(slides[1],72,480,1140,70,'Setup 3: Strong E1R30 versus E3R10 at equal exposure',31);
text(slides[1],72,570,1140,75,'5 configurations x 3 seeds. 1,620,000 image exposures per run.',30);
const values=[['Setting','Accuracy (%)','Macro-F1'],...settings.map((s,i)=>[labels[i],`${(100*Number(summaries[s].accuracy_mean)).toFixed(2)} +/- ${(100*Number(summaries[s].accuracy_sd)).toFixed(2)}`,Number(summaries[s].macro_f1_mean).toFixed(4)])];
const table=slides[2].tables.add({rows:6,columns:3,left:72,top:150,width:710,height:370,columnWidths:[285,250,175],values});
table.cells.block({row:0,column:0,rowCount:6,columnCount:3}).assign({textStyle:{typeface:'Arial',fontSize:25,color:'#192e55'},fill:'#ffffff',margins:{left:8,right:8,top:12,bottom:12}});
table.cells.block({row:0,column:0,rowCount:1,columnCount:3}).assign({textStyle:{typeface:'Arial',fontSize:25,bold:true,color:'#192e55'},fill:'#eef2f7'});
table.borders.assign({width:0,fill:'#ffffff'});
text(slides[2],825,175,385,130,`Strong skew: ${gap} pp\nversus IID`,33,true);
text(slides[2],825,350,385,190,`E3R10: +${delta} pp\nversus E1R30 with\n20 fewer rounds`,32,true);
text(slides[2],80,575,1130,65,'Final checkpoints. Mean +/- sample SD over three seeds.',28);
text(slides[3],72,155,650,100,'Label skew reduces global quality in this protocol.',30);
text(slides[3],72,285,650,120,'Fewer rounds retain similar observed mean accuracy at fixed exposure.',30);
text(slides[3],72,435,650,85,'Limits: one dataset/CNN, three seeds, simulated clients.',30);
text(slides[3],72,545,650,100,'Demo: strong seed 42, one test prediction from a trained checkpoint.',30);
slides[3].images.add({blob:new Uint8Array(await fs.readFile(path.join(repoRoot,'reports/figures/07_error_examples.png'))),contentType:'image/png',alt:'Eight actual Shirt-to-T-shirt/top errors in strong seed 42',fit:'contain',position:{left:780,top:190,width:435,height:275}});
text(slides[3],790,500,425,65,`Shirt recall: ${shirtRecall.toFixed(4)}`,27);
text(slides[3],790,565,425,90,`Shirt to T-shirt/top: ${metrics.confusion_matrix[6][0]} errors (seed 42)`,27);
const notes=[
'0:00–0:40. We study image classification with unequal client label proportions. Ten clients retain equal quantities. Our questions concern the centralized gap, label skew and aggregation frequency. Sources: project report Section I; FedAvg https://proceedings.mlr.press/v54/mcmahan17a.html .',
'0:40–1:25. One CNN and a fixed split support all three setups. Each client receives the same global snapshot. Setup 1 compares centralized and IID; Setup 2 varies skew; Setup 3 compares E1R30 and E3R10 at fixed exposure. Dataset: https://github.com/zalandoresearch/fashion-mnist . Evidence: DATA.md and configs.',
'1:25–2:15. The table shows final checkpoints, mean and sample SD across seeds 42/43/44. Strong loses 5.52 pp versus IID. E3 differs by +0.04 pp, without proof of equivalence. Source: outputs/tables/summary.csv and the fifteen final_metrics.json files.',
'2:15–2:55. The result applies to this dataset, CNN and simulation. Client drift is plausible but unmeasured. Show one pretrained strong seed-42 prediction; stop before three minutes. All members prepare for the following twelve-minute Q/A. Image: Fashion-MNIST, Zalando Research, MIT; recorded test indices 40,145,226,269,286,293,322,344. Source: outputs/tables/error_examples.csv. Drift context: https://proceedings.mlr.press/v119/karimireddy20a.html .'
];
slides.forEach((slide,i)=>slide.speakerNotes.textFrame.setText(notes[i]));
const candidatePath=path.join(buildDir,'candidate.pptx');
await (await PresentationFile.exportPptx(p)).save(candidatePath);
if(process.env.SKILL_DIR){
 const {finalizePresentation}=await import(pathToFileURL(path.join(process.env.SKILL_DIR,'container_tools/artifact_tool_utils.mjs')).href);
 await finalizePresentation({workspaceDir:process.env.WORKSPACE_DIR,candidatePath,finalPath:process.env.FINAL_PPTX,
 pythonExecutable:process.env.RUNTIME_PYTHON,
 integrityValidatorPath:path.join(process.env.SKILL_DIR,'container_tools/inspect_presentation_package_integrity.py'),
 layoutValidatorPath:path.join(process.env.SKILL_DIR,'container_tools/inspect_presentation_layout_geometry.py'),
 layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-heading-fit','--require-native-table-slide','3'],
 explicitTotalSlideCount:4,requiredNativeTableOwnerSlides:[3],requiredNativeChartOwnerSlides:[],fontPolicy:{basis:'design',families:['Arial']},verifyArtifactToolImport:true,receiptPath:path.join(buildDir,'validation.json')});
}else console.log(`Draft written to ${candidatePath}`);
