export interface CitationPayload {
  doc_name: string;
  page: number;
}

export interface NodeEvent {
  id: number;
  type: "node";
  label: string;
}

export interface TokenEvent {
  id: number;
  type: "token";
  text: string;
}

export interface CitationsEvent {
  id: number;
  type: "citations";
  citations: CitationPayload[];
}

export interface FinalEvent {
  id: number;
  type: "final";
  answer: string;
  abstained: boolean;
  cache_hit: boolean;
}

export interface ErrorEvent {
  id: number;
  type: "error";
  message: "The request could not be completed. Please try again.";
}

export type ServerEvent =
  | NodeEvent
  | TokenEvent
  | CitationsEvent
  | FinalEvent
  | ErrorEvent;
