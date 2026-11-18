export type VaRow = {
  t: number;
  v: number;
  a: number;
  dominance: string;
  token: string;
};

/** Minimal RFC 4180 parser: tokens contain leading spaces and quoted commas. */
function parseCsv(text: string): string[][] {
  const rows: string[][] = [];
  let row: string[] = [];
  let field = "";
  let quoted = false;
  for (let i = 0; i < text.length; i += 1) {
    const ch = text[i];
    if (quoted) {
      if (ch === '"' && text[i + 1] === '"') {
        field += '"';
        i += 1;
      } else if (ch === '"') {
        quoted = false;
      } else {
        field += ch;
      }
    } else if (ch === '"') {
      quoted = true;
    } else if (ch === ",") {
      row.push(field);
      field = "";
    } else if (ch === "\n" || ch === "\r") {
      if (ch === "\r" && text[i + 1] === "\n") i += 1;
      row.push(field);
      rows.push(row);
      row = [];
      field = "";
    } else {
      field += ch;
    }
  }
  if (field !== "" || row.length > 0) {
    row.push(field);
    rows.push(row);
  }
  return rows;
}

export function parseVaCsv(text: string): VaRow[] {
  const [header, ...body] = parseCsv(text);
  const col = (name: string) => header.indexOf(name);
  const [it, iv, ia, id, itok] = ["t", "v", "a", "dominance", "token"].map(col);
  return body
    .filter((r) => r.length >= header.length)
    .map((r) => ({
      t: Number(r[it]),
      v: Number(r[iv]),
      a: Number(r[ia]),
      dominance: id >= 0 ? r[id] : "",
      token: itok >= 0 ? r[itok] : "",
    }));
}

function quote(field: string): string {
  return /[",\n\r]/.test(field) ? `"${field.replace(/"/g, '""')}"` : field;
}

export function toVaCsv(rows: VaRow[]): string {
  const lines = ["t,v,a,dominance,token"];
  for (const r of rows) {
    lines.push([String(r.t), String(r.v), String(r.a), r.dominance, quote(r.token)].join(","));
  }
  return `${lines.join("\n")}\n`;
}
