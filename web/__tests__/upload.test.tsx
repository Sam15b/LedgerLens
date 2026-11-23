import { act, fireEvent, render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import { Upload } from "@/components/Upload";
import { checkFile, type UploadOutcome } from "@/lib/upload";

const refresh = vi.fn();
vi.mock("next/navigation", () => ({ useRouter: () => ({ refresh }) }));

const upload = vi.hoisted(() => ({
  pending: [] as {
    file: File;
    onProgress?: (share: number) => void;
    finish: (outcome: UploadOutcome) => void;
  }[],
}));
vi.mock("@/lib/upload", async (importOriginal) => {
  const real = await importOriginal<typeof import("@/lib/upload")>();
  return {
    ...real,
    uploadFile: (file: File, options: { onProgress?: (share: number) => void } = {}) =>
      new Promise<UploadOutcome>((finish) => {
        upload.pending.push({ file, onProgress: options.onProgress, finish });
      }),
  };
});

function pdf(name = "invoice.pdf", size = 2048): File {
  return new File([new Uint8Array(size)], name, { type: "application/pdf" });
}

function drop(...files: File[]) {
  fireEvent.drop(screen.getByTestId("dropzone"), { dataTransfer: { files } });
}

beforeEach(() => {
  upload.pending.length = 0;
  refresh.mockClear();
});

describe("Upload", () => {
  it("shows progress while a dropped file uploads, then links to its document", async () => {
    render(<Upload />);
    drop(pdf());
    expect(upload.pending).toHaveLength(1);
    const bar = screen.getByRole("progressbar", { name: "Uploading invoice.pdf" });
    expect(bar.getAttribute("aria-valuenow")).toBe("0");

    act(() => upload.pending[0]!.onProgress?.(0.42));
    expect(bar.getAttribute("aria-valuenow")).toBe("42");
    expect(screen.getByText("42%")).toBeTruthy();

    await act(async () =>
      upload.pending[0]!.finish({
        ok: true,
        body: { document_id: "doc-1", status: "queued", duplicate: false, job_id: "job-1" },
      }),
    );
    expect(screen.queryByRole("progressbar")).toBeNull();
    expect(screen.getByText(/Uploaded and queued for ingestion/)).toBeTruthy();
    expect(screen.getByRole("link", { name: "Open the document" }).getAttribute("href")).toBe(
      "/documents/doc-1",
    );
    expect(refresh).toHaveBeenCalledTimes(1);
  });

  it("uploads several files one after another and reports a duplicate", async () => {
    render(<Upload />);
    drop(pdf("a.pdf"), pdf("b.pdf"));
    expect(upload.pending.map((p) => p.file.name)).toEqual(["a.pdf"]);
    await act(async () =>
      upload.pending[0]!.finish({
        ok: true,
        body: { document_id: "doc-a", status: "ready", duplicate: true },
      }),
    );
    expect(upload.pending.map((p) => p.file.name)).toEqual(["a.pdf", "b.pdf"]);
    expect(screen.getByText(/Already uploaded/)).toBeTruthy();
  });

  it("shows the API's refusal", async () => {
    render(<Upload />);
    drop(pdf("notes.pdf"));
    await act(async () =>
      upload.pending[0]!.finish({
        ok: false,
        status: 415,
        message: "unsupported file type text/plain (request abc)",
      }),
    );
    expect(screen.getByRole("alert").textContent).toBe(
      "unsupported file type text/plain (request abc)",
    );
    expect(refresh).not.toHaveBeenCalled();
  });

  it("refuses an empty file without sending it", () => {
    render(<Upload />);
    drop(pdf("empty.pdf", 0));
    expect(upload.pending).toHaveLength(0);
    expect(screen.getByRole("alert").textContent).toBe("The file is empty.");
  });

  it("takes files from the picker too", () => {
    render(<Upload />);
    fireEvent.change(screen.getByLabelText("Files to upload"), { target: { files: [pdf()] } });
    expect(upload.pending).toHaveLength(1);
  });
});

describe("checkFile", () => {
  it("allows files up to the limit", () => {
    expect(checkFile({ size: 1 })).toBeNull();
    expect(checkFile({ size: 25 * 1024 * 1024 })).toBeNull();
    expect(checkFile({ size: 25 * 1024 * 1024 + 1 })).toMatch(/25 MB limit/);
  });
});

/** Just enough of XMLHttpRequest for uploadFile. */
class FakeXhr {
  static last: FakeXhr | null = null;
  method = "";
  url = "";
  responseType = "";
  status = 0;
  response: unknown = null;
  sent: FormData | null = null;
  upload: { onprogress: ((e: Partial<ProgressEvent>) => void) | null } = { onprogress: null };
  onload: (() => void) | null = null;
  onerror: (() => void) | null = null;
  onabort: (() => void) | null = null;

  constructor() {
    FakeXhr.last = this;
  }
  open(method: string, url: string) {
    this.method = method;
    this.url = url;
  }
  send(body: FormData) {
    this.sent = body;
  }
  abort() {
    this.onabort?.();
  }
  answer(status: number, response: unknown) {
    this.status = status;
    this.response = response;
    this.onload?.();
  }
}

describe("uploadFile", () => {
  it("posts the file as multipart and reports progress", async () => {
    const real = await vi.importActual<typeof import("@/lib/upload")>("@/lib/upload");
    vi.stubGlobal("XMLHttpRequest", FakeXhr);
    const shares: number[] = [];
    const pending = real.uploadFile(pdf(), {
      baseUrl: "http://api.test",
      onProgress: (s) => shares.push(s),
    });
    const xhr = FakeXhr.last!;
    expect([xhr.method, xhr.url]).toEqual(["POST", "http://api.test/documents"]);
    expect((xhr.sent?.get("file") as File).name).toBe("invoice.pdf");

    xhr.upload.onprogress?.({ lengthComputable: true, loaded: 512, total: 2048 });
    xhr.answer(202, { document_id: "d", status: "queued", duplicate: false, job_id: "j" });
    await expect(pending).resolves.toEqual({
      ok: true,
      body: { document_id: "d", status: "queued", duplicate: false, job_id: "j" },
    });
    expect(shares).toEqual([0.25, 1]);
    vi.unstubAllGlobals();
  });

  it("turns an error body, a network failure and a cancel into messages", async () => {
    const real = await vi.importActual<typeof import("@/lib/upload")>("@/lib/upload");
    vi.stubGlobal("XMLHttpRequest", FakeXhr);

    const refused = real.uploadFile(pdf(), { baseUrl: "" });
    FakeXhr.last!.answer(413, {
      error: { code: "too_large", message: "the upload is over the 25 MB limit", request_id: "r1" },
    });
    await expect(refused).resolves.toEqual({
      ok: false,
      status: 413,
      message: "the upload is over the 25 MB limit (request r1)",
    });

    const offline = real.uploadFile(pdf(), { baseUrl: "" });
    FakeXhr.last!.onerror?.();
    await expect(offline).resolves.toMatchObject({ ok: false, status: null });

    const controller = new AbortController();
    const cancelled = real.uploadFile(pdf(), { baseUrl: "", signal: controller.signal });
    controller.abort();
    await expect(cancelled).resolves.toMatchObject({ message: "The upload was cancelled." });
    vi.unstubAllGlobals();
  });
});
