import { app } from '../../scripts/app.js';

app.registerExtension({
    name: 'H3.SemanticTimeline.Preview',
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== 'H3SemanticTimelinePreview') return;
        const original = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            original?.apply(this, arguments);
            if (!this.h3Preview) {
                const area = document.createElement('textarea');
                area.readOnly = true;
                area.setAttribute('aria-label', 'H3 semantic timeline preview');
                area.style.cssText = 'width:100%;height:100%;min-height:180px;box-sizing:border-box;resize:none;font:12px monospace;background:#171717;color:#eee;';
                this.addDOMWidget('h3_timeline_preview', 'text', area, { serialize: false });
                this.h3Preview = area;
                this.setSize([Math.max(this.size[0], 560), Math.max(this.size[1], 620)]);
            }
            this.h3Preview.value = (message.text || []).join('\n\n');
            app.graph.setDirtyCanvas(true, true);
        };
    },
});
