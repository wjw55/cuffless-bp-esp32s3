"use strict";

// Execute the production sizing function, with canvas attribute reflection and
// integer backing dimensions, without a browser or third-party dependencies.
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");

const assets = path.join(__dirname, "..", "dashboard_assets");
const source = fs.readFileSync(path.join(assets, "dashboard.js"), "utf8");
const sizingFunction = source.match(/^function resizeCanvas\(canvas\) \{[\s\S]*?^\}/m);
assert.ok(sizingFunction, "Production resizeCanvas function must be present");

function makeCanvas(height, clientWidth = 801) {
  const context = {
    transform: null,
    setTransform(...values) { this.transform = values; },
  };
  const canvas = {
    clientWidth,
    dataset: {chartHeight: String(height)},
    style: {},
    sizeWrites: 0,
    _width: 300,
    _height: height,
    get width() { return this._width; },
    set width(value) { this._width = Math.trunc(value); this.sizeWrites += 1; },
    get height() { return this._height; },
    set height(value) { this._height = Math.trunc(value); this.sizeWrites += 1; },
    getAttribute(name) { return String(this[name]); },
    getContext(kind) { assert.equal(kind, "2d"); return context; },
  };
  return {canvas, context};
}

function makeResize(ratio) {
  const window = {devicePixelRatio: ratio};
  const resize = vm.runInNewContext(`(${sizingFunction[0]})`, {window});
  return {resize, window};
}

function checkSize(resize, canvas, height, ratio) {
  const result = resize(canvas);
  const width = Math.max(200, canvas.clientWidth);
  assert.equal(result.width, width);
  assert.equal(result.height, height);
  assert.equal(canvas.width, Math.round(width * ratio));
  assert.equal(canvas.height, Math.round(height * ratio));
  assert.equal(canvas.style.height, `${height}px`);
  assert.deepEqual(Array.from(result.ctx.transform), [ratio, 0, 0, ratio, 0, 0]);
}

const cases = {
  repeated() {
    for (const ratio of [1, 1.25, 1.5, 2, 3]) {
      for (const height of [220, 250]) {
        const {canvas} = makeCanvas(height);
        const {resize} = makeResize(ratio);
        checkSize(resize, canvas, height, ratio);
        const writes = canvas.sizeWrites;
        for (let update = 0; update < 1000; update += 1) {
          checkSize(resize, canvas, height, ratio);
        }
        assert.equal(canvas.sizeWrites, writes, "Stable charts must not reallocate on every update");
      }
    }
  },
  resize() {
    for (const height of [220, 250]) {
      const {canvas} = makeCanvas(height);
      const {resize, window} = makeResize(1);
      for (const ratio of [1.25, 2, 1.5, 1]) {
        window.devicePixelRatio = ratio;
        for (const width of [801, 420, 0, 1023]) {
          canvas.clientWidth = width;
          checkSize(resize, canvas, height, ratio);
          const writes = canvas.sizeWrites;
          checkSize(resize, canvas, height, ratio);
          assert.equal(canvas.sizeWrites, writes);
        }
      }
    }
  },
  oversized() {
    for (const height of [220, 250]) {
      const {canvas} = makeCanvas(height);
      const {resize} = makeResize(2);
      canvas.height = 901120;
      checkSize(resize, canvas, height, 2);
    }
    const {canvas} = makeCanvas(220);
    delete canvas.dataset.chartHeight;
    canvas.height = 440;
    checkSize(makeResize(2).resize, canvas, 220, 2);
  },
  ratio() {
    for (const ratio of [undefined, 0, -1, NaN, Infinity]) {
      const {canvas} = makeCanvas(220);
      checkSize(makeResize(ratio).resize, canvas, 220, 1);
    }
  },
  markup() {
    const html = fs.readFileSync(path.join(assets, "index.html"), "utf8");
    for (const [id, height] of [["waveform", 220], ["history", 250]]) {
      const tag = html.match(new RegExp(`<canvas\\b[^>]*\\bid="${id}"[^>]*>`));
      assert.ok(tag, `${id} canvas must exist`);
      assert.match(tag[0], new RegExp(`\\bdata-chart-height="${height}"`));
      assert.match(tag[0], new RegExp(`\\bheight="${height}"`));
    }
  },
};

const requested = process.argv[2];
assert.ok(!requested || Object.hasOwn(cases, requested), "Unknown canvas test case");
for (const name of requested ? [requested] : Object.keys(cases)) {
  cases[name]();
  console.log(`PASS ${name}`);
}
