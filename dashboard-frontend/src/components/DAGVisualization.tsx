import { useCallback, useMemo, useState } from "react";
import {
  ReactFlow,
  Background,
  Controls,
  MiniMap,
  useNodesState,
  useEdgesState,
  type Node,
  type Edge,
  type NodeProps,
  Handle,
  Position,
  MarkerType,
  Panel,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import type { DAGNode, DAGEdge, DAGExperiment } from "../api/models";
import { Badge } from "./ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "./ui/card";
import { Link } from "@tanstack/react-router";

interface DAGVisualizationProps {
  nodes: DAGNode[];
  edges: DAGEdge[];
}

// Type for node data in React Flow
type ClassNodeData = DAGNode & Record<string, unknown>;

// Custom node component for displaying class information
function ClassNode({ data, selected }: NodeProps<Node<ClassNodeData>>) {
  const node = data;
  const hasFailures = node.failed_count > 0;
  const hasRunning = node.running_count > 0;
  const allSuccess = node.success_count === node.total_count && node.total_count > 0;

  // Determine border color based on status
  let borderColor = "border-border";
  if (selected) {
    borderColor = "border-primary ring-2 ring-primary/20";
  } else if (hasFailures) {
    borderColor = "border-red-500";
  } else if (hasRunning) {
    borderColor = "border-blue-500";
  } else if (allSuccess) {
    borderColor = "border-green-500";
  }

  return (
    <div
      className={`bg-card rounded-lg border-2 ${borderColor} shadow-md min-w-[180px] transition-all`}
    >
      <Handle
        type="target"
        position={Position.Top}
        className="!bg-muted-foreground !w-3 !h-3"
      />

      <div className="px-4 py-3">
        <div className="font-semibold text-sm mb-2">{node.class_name}</div>

        {/* Experiment counts */}
        <div className="flex flex-wrap gap-1.5 mb-2">
          <Badge variant="muted" className="text-xs px-1.5 py-0.5">
            {node.total_count} total
          </Badge>
          {node.success_count > 0 && (
            <Badge variant="success" className="text-xs px-1.5 py-0.5">
              {node.success_count}
            </Badge>
          )}
          {node.running_count > 0 && (
            <Badge variant="info" className="text-xs px-1.5 py-0.5">
              {node.running_count}
            </Badge>
          )}
          {node.failed_count > 0 && (
            <Badge variant="destructive" className="text-xs px-1.5 py-0.5">
              {node.failed_count}
            </Badge>
          )}
        </div>

        {/* Mini experiment list (show first 3) */}
        <div className="space-y-1">
          {node.experiments.slice(0, 3).map((exp: DAGExperiment) => (
            <Link
              key={exp.huldra_hash}
              to="/experiments/$namespace/$huldra_hash"
              params={{
                namespace: exp.namespace,
                huldra_hash: exp.huldra_hash,
              }}
              className="block text-xs text-muted-foreground hover:text-primary truncate"
              title={exp.huldra_hash}
            >
              {exp.huldra_hash.slice(0, 8)}...
            </Link>
          ))}
          {node.experiments.length > 3 && (
            <div className="text-xs text-muted-foreground">
              +{node.experiments.length - 3} more
            </div>
          )}
        </div>
      </div>

      <Handle
        type="source"
        position={Position.Bottom}
        className="!bg-muted-foreground !w-3 !h-3"
      />
    </div>
  );
}

const nodeTypes = {
  classNode: ClassNode,
};

// Layout algorithm: topological sort with horizontal positioning
function layoutNodes(
  dagNodes: DAGNode[],
  dagEdges: DAGEdge[]
): { nodes: Node<ClassNodeData>[]; edges: Edge[] } {
  // Build adjacency lists
  const inDegree = new Map<string, number>();
  const outEdges = new Map<string, string[]>();
  const nodeMap = new Map<string, DAGNode>();

  for (const node of dagNodes) {
    inDegree.set(node.id, 0);
    outEdges.set(node.id, []);
    nodeMap.set(node.id, node);
  }

  for (const edge of dagEdges) {
    // Only count edges where both source and target exist
    if (nodeMap.has(edge.source) && nodeMap.has(edge.target)) {
      inDegree.set(edge.target, (inDegree.get(edge.target) ?? 0) + 1);
      outEdges.get(edge.source)?.push(edge.target);
    }
  }

  // Topological sort to determine layers
  const layers: string[][] = [];
  const assigned = new Set<string>();
  const remaining = new Set(dagNodes.map((n) => n.id));

  while (remaining.size > 0) {
    // Find all nodes with in-degree 0 from remaining nodes
    const currentLayer: string[] = [];
    for (const nodeId of remaining) {
      const degree = inDegree.get(nodeId) ?? 0;
      if (degree === 0) {
        currentLayer.push(nodeId);
      }
    }

    // If no nodes have in-degree 0, there's a cycle - just pick one
    if (currentLayer.length === 0) {
      const first = remaining.values().next().value;
      if (first) currentLayer.push(first);
    }

    // Add current layer
    layers.push(currentLayer);

    // Remove current layer nodes and update in-degrees
    for (const nodeId of currentLayer) {
      remaining.delete(nodeId);
      assigned.add(nodeId);
      for (const targetId of outEdges.get(nodeId) ?? []) {
        if (!assigned.has(targetId)) {
          inDegree.set(targetId, (inDegree.get(targetId) ?? 1) - 1);
        }
      }
    }
  }

  // Position nodes
  const NODE_WIDTH = 200;
  const LAYER_SPACING = 180;
  const NODE_SPACING = 40;

  const nodePositions = new Map<string, { x: number; y: number }>();

  for (let layerIdx = 0; layerIdx < layers.length; layerIdx++) {
    const layer = layers[layerIdx];
    const layerWidth = layer.length * NODE_WIDTH + (layer.length - 1) * NODE_SPACING;
    const startX = -layerWidth / 2;

    for (let nodeIdx = 0; nodeIdx < layer.length; nodeIdx++) {
      const nodeId = layer[nodeIdx];
      nodePositions.set(nodeId, {
        x: startX + nodeIdx * (NODE_WIDTH + NODE_SPACING),
        y: layerIdx * LAYER_SPACING,
      });
    }
  }

  // Create React Flow nodes
  const flowNodes: Node<ClassNodeData>[] = dagNodes.map((node) => {
    const position = nodePositions.get(node.id) ?? { x: 0, y: 0 };
    return {
      id: node.id,
      type: "classNode",
      position,
      data: node as ClassNodeData,
    };
  });

  // Create React Flow edges
  const flowEdges: Edge[] = dagEdges
    .filter((e) => nodeMap.has(e.source) && nodeMap.has(e.target))
    .map((edge) => ({
      id: `${edge.source}-${edge.target}`,
      source: edge.source,
      target: edge.target,
      label: edge.field_name,
      labelStyle: { fontSize: 10, fill: "hsl(var(--muted-foreground))" },
      labelBgStyle: { fill: "hsl(var(--background))" },
      labelBgPadding: [4, 2] as [number, number],
      style: { stroke: "hsl(var(--border))", strokeWidth: 2 },
      markerEnd: {
        type: MarkerType.ArrowClosed,
        color: "hsl(var(--border))",
      },
      animated: false,
    }));

  return { nodes: flowNodes, edges: flowEdges };
}

export function DAGVisualization({ nodes: dagNodes, edges: dagEdges }: DAGVisualizationProps) {
  const [selectedNode, setSelectedNode] = useState<DAGNode | null>(null);

  // Layout and convert to React Flow format
  const { nodes: initialNodes, edges: initialEdges } = useMemo(
    () => layoutNodes(dagNodes, dagEdges),
    [dagNodes, dagEdges]
  );

  const [nodes, , onNodesChange] = useNodesState(initialNodes);
  const [edges, setEdges, onEdgesChange] = useEdgesState(initialEdges);

  // Highlight connected nodes on click
  const onNodeClick = useCallback(
    (_: React.MouseEvent, node: Node) => {
      const nodeData = node.data as ClassNodeData;
      setSelectedNode(nodeData);

      // Find connected nodes (upstream and downstream)
      const connectedIds = new Set<string>([node.id]);

      // Find upstream (sources that connect to this node)
      for (const edge of dagEdges) {
        if (edge.target === node.id) {
          connectedIds.add(edge.source);
        }
        if (edge.source === node.id) {
          connectedIds.add(edge.target);
        }
      }

      // Update edge styles
      setEdges((eds) =>
        eds.map((e) => {
          const isConnected = e.source === node.id || e.target === node.id;
          return {
            ...e,
            animated: isConnected,
            style: {
              ...e.style,
              stroke: isConnected ? "hsl(var(--primary))" : "hsl(var(--border))",
              strokeWidth: isConnected ? 3 : 2,
            },
            markerEnd: {
              type: MarkerType.ArrowClosed,
              color: isConnected ? "hsl(var(--primary))" : "hsl(var(--border))",
            },
          };
        })
      );
    },
    [dagEdges, setEdges]
  );

  // Reset highlighting when clicking on background
  const onPaneClick = useCallback(() => {
    setSelectedNode(null);
    setEdges((eds) =>
      eds.map((e) => ({
        ...e,
        animated: false,
        style: { ...e.style, stroke: "hsl(var(--border))", strokeWidth: 2 },
        markerEnd: {
          type: MarkerType.ArrowClosed,
          color: "hsl(var(--border))",
        },
      }))
    );
  }, [setEdges]);

  return (
    <div className="h-[600px] border rounded-lg bg-background relative">
      <ReactFlow
        nodes={nodes}
        edges={edges}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        onNodeClick={onNodeClick}
        onPaneClick={onPaneClick}
        nodeTypes={nodeTypes}
        fitView
        fitViewOptions={{ padding: 0.2 }}
        minZoom={0.1}
        maxZoom={2}
        defaultEdgeOptions={{
          type: "smoothstep",
        }}
      >
        <Background />
        <Controls />
        <MiniMap
          nodeStrokeColor="hsl(var(--border))"
          nodeColor="hsl(var(--card))"
          nodeBorderRadius={4}
        />

        {/* Selected node details panel */}
        {selectedNode && (
          <Panel position="top-right" className="m-4">
            <Card className="w-80">
              <CardHeader className="pb-2">
                <CardTitle className="text-base">{selectedNode.class_name}</CardTitle>
                <code className="text-xs text-muted-foreground break-all">
                  {selectedNode.full_class_name}
                </code>
              </CardHeader>
              <CardContent className="space-y-3">
                <div className="flex gap-2 flex-wrap">
                  <Badge variant="muted">{selectedNode.total_count} experiments</Badge>
                  {selectedNode.success_count > 0 && (
                    <Badge variant="success">{selectedNode.success_count} success</Badge>
                  )}
                  {selectedNode.running_count > 0 && (
                    <Badge variant="info">{selectedNode.running_count} running</Badge>
                  )}
                  {selectedNode.failed_count > 0 && (
                    <Badge variant="destructive">{selectedNode.failed_count} failed</Badge>
                  )}
                </div>

                <div className="max-h-40 overflow-y-auto space-y-1">
                  {selectedNode.experiments.map((exp) => (
                    <Link
                      key={exp.huldra_hash}
                      to="/experiments/$namespace/$huldra_hash"
                      params={{
                        namespace: exp.namespace,
                        huldra_hash: exp.huldra_hash,
                      }}
                      className="flex items-center justify-between p-2 rounded hover:bg-muted text-sm"
                    >
                      <code className="text-xs">{exp.huldra_hash.slice(0, 12)}...</code>
                      <Badge
                        variant={
                          exp.result_status === "success"
                            ? "success"
                            : exp.result_status === "failed"
                              ? "destructive"
                              : exp.attempt_status === "running"
                                ? "info"
                                : "muted"
                        }
                        className="text-xs"
                      >
                        {exp.attempt_status ?? exp.result_status}
                      </Badge>
                    </Link>
                  ))}
                </div>
              </CardContent>
            </Card>
          </Panel>
        )}
      </ReactFlow>
    </div>
  );
}
