import { useEffect, useState } from "react";
import { ApiError, api } from "../api";
import { Banner, Card, Pill, Stat } from "../components/ui";
import type { Roadmap, RoadmapItem } from "../types";

const STATUS: Record<RoadmapItem["status"], { label: string; tone: "ok" | "warn" | "neutral" }> = {
  done: { label: "已完成", tone: "ok" },
  partial: { label: "进行中", tone: "warn" },
  todo: { label: "待做", tone: "neutral" }
};

export function RoadmapPage() {
  const [roadmap, setRoadmap] = useState<Roadmap | null>(null);
  const [error, setError] = useState("");

  useEffect(() => {
    api
      .roadmap()
      .then(setRoadmap)
      .catch((cause) => setError(cause instanceof ApiError ? cause.message : String(cause)));
  }, []);

  if (error) return <Banner tone="error">{error}</Banner>;
  if (!roadmap) return <div className="empty">加载中 ...</div>;

  return (
    <div className="grid">
      <Card title="总览" padded={false}>
        <div className="stats">
          <Stat label="已完成" value={roadmap.counts.done ?? 0} />
          <Stat label="进行中" value={roadmap.counts.partial ?? 0} />
          <Stat label="待做" value={roadmap.counts.todo ?? 0} />
          <Stat label="合计" value={roadmap.total} />
        </div>
        <div className="body small muted">
          这张表由 <span className="mono">src/so_snake/gui/roadmap.py</span> 提供,和代码放在一起,
          所以落地一个模块的人顺手就能改对。「已完成」的意思是有代码也有 gate 覆盖;
          能量化的结论写在证据一栏。
        </div>
      </Card>

      {roadmap.groups.map((group) => (
        <Card key={group.group} title={group.group} padded={false}>
          {group.items.map((item) => (
            <div className="roadmap-item" key={item.key}>
              <div>
                <Pill tone={STATUS[item.status].tone}>{STATUS[item.status].label}</Pill>
              </div>
              <div>
                <div className="title">{item.title}</div>
                <div className="detail">{item.detail}</div>
                <div className="meta">
                  {item.module ? <span>{item.module}</span> : null}
                  {item.evidence ? <span> · 证据: {item.evidence}</span> : null}
                  {item.blockers.length ? <span> · 卡在: {item.blockers.join(", ")}</span> : null}
                </div>
              </div>
            </div>
          ))}
        </Card>
      ))}
    </div>
  );
}
