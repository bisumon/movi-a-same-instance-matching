import fs from 'node:fs/promises';
import { FileBlob, SpreadsheetFile } from '@oai/artifact-tool';

const [moviAPath, moviDEPath, outDir] = process.argv.slice(2);
await fs.mkdir(outDir, { recursive: true });

for (const [label, inputPath] of [['movi_a', moviAPath], ['movi_de', moviDEPath]]) {
  const workbook = await SpreadsheetFile.importXlsx(await FileBlob.load(inputPath));
  const summary = await workbook.inspect({
    kind: 'workbook,sheet,table',
    include: 'id,name,values,formulas',
    maxChars: 18000,
    tableMaxRows: 30,
    tableMaxCols: 20,
    tableMaxCellChars: 120,
  });
  await fs.writeFile(`${outDir}/${label}_inspect.ndjson`, summary.ndjson);
  const sheetInfo = await workbook.inspect({ kind: 'sheet', include: 'id,name', maxChars: 12000 });
  await fs.writeFile(`${outDir}/${label}_sheets.ndjson`, sheetInfo.ndjson);
  const names = [...sheetInfo.ndjson.matchAll(/"name":"([^"]+)"/g)].map((match) => match[1]);
  for (let i = 0; i < names.length; i += 1) {
    const rendered = await workbook.render({ sheetName: names[i], autoCrop: 'all', scale: 1, format: 'png' });
    const safeName = names[i].replaceAll(/[^A-Za-z0-9_-]+/g, '_');
    await fs.writeFile(`${outDir}/${label}_${String(i + 1).padStart(2, '0')}_${safeName}.png`, new Uint8Array(await rendered.arrayBuffer()));
  }
}
