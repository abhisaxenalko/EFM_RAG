export type CitationPayload = {
  doc_name: string;
  page: number;
};

export type ServerEvent =
  | { id: number; type: "node"; label: string }
  | { id: number; type: "token"; text: string }
  | { id: number; type: "citations"; citations: CitationPayload[] }
  | {
      id: number;
      type: "final";
      answer: string;
      abstained: boolean;
      cache_hit: boolean;
    }
  | {
      id: number;
      type: "error";
      message: "The request could not be completed. Please try again.";
    };
