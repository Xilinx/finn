/****************************************************************************
 * Copyright Advanced Micro Devices, Inc.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * @brief Crop the spatial borders from a folded feature-map stream.
 * @author Oliver Cassidy <oliver.cassidy@amd.com>
 * @author Thomas B. Preußer <thomas.preusser@amd.com>
 *
 * @description
 * Consumes feature maps ordered by height, width, and channel fold. Folds
 * inside the retained rectangle are forwarded; border folds are consumed
 * and discarded. A two-entry forwarding path absorbs output backpressure
 * while allowing discarded input folds to continue advancing.
 ***************************************************************************/

module crop #(
    int unsigned H,
    int unsigned W,
    int unsigned CF,
    int unsigned DATA_WIDTH,
    int unsigned CROP_N,
    int unsigned CROP_E = CROP_N,
    int unsigned CROP_S = CROP_N,
    int unsigned CROP_W = CROP_E
)(
    // Global Control
    input logic clk,
    input logic rst,

    // Input Stream
    output logic irdy,
    input logic ivld,
    input logic [DATA_WIDTH-1:0] idat,

    // Output Stream
    input logic ordy,
    output logic ovld,
    output logic [DATA_WIDTH-1:0] odat
);

    //=== Parameter Validation ==============================================
    initial begin
        automatic bit fail = 0;

        if(H < 1) begin
            $error("%m: H must be at least 1.");
            fail = 1;
        end
        if(W < 1) begin
            $error("%m: W must be at least 1.");
            fail = 1;
        end
        if(CF < 1) begin
            $error("%m: CF must be at least 1.");
            fail = 1;
        end
        if(DATA_WIDTH < 1) begin
            $error("%m: DATA_WIDTH must be at least 1.");
            fail = 1;
        end
        if(CROP_N + CROP_S >= H) begin
            $error("%m: North and south cropping must leave at least one row.");
            fail = 1;
        end
        if(CROP_W + CROP_E >= W) begin
            $error("%m: West and east cropping must leave at least one column.");
            fail = 1;
        end
        if(fail) $finish;
    end

    typedef logic [DATA_WIDTH-1:0] data_t;

    //=== Feature-Map Position ==============================================

    // Channel Fold (degenerates to constant for CF=1)
    logic signed [$clog2(CF):0] CFCnt = CF - 2;  // CF-2, .., 1, 0, -1 (last)
    uwire cf_last = CFCnt[$left(CFCnt)];
    always_ff @(posedge clk) begin
        if(rst) CFCnt <= CF - 2;
        else if(take) CFCnt <= CFCnt - (cf_last? 1-CF : 1);
    end

    typedef struct {
        logic act;
        logic post;
    } phase_t;

    // Width
    uwire wlst;
    uwire wact;
    if(W < 2) begin : genWTrivial
        assign wlst = 1;
        assign wact = 1;
    end : genWTrivial
    else begin : genWCnt
        //                 | <------ active ------> |
        // 0, ..., CROP_W-1, CROP_W, ..., W-CROP_E-1, W-CROP_E, ..., W-1 (last)
        logic [$clog2(W)-1:0] WCnt = 0;
        logic WLst = 0;
        phase_t WPhase = '{ act: CROP_W == 0, post: 0 };
        always_ff @(posedge clk) begin
            if(rst) begin
                WCnt <= 0;
                WLst <= 0;
                WPhase <= '{ act: CROP_W == 0, post: 0 };
            end
            else if(take && cf_last) begin
                WCnt <= WCnt + (WLst? -W+1 : 1);
                WLst <= !((W - 2) & ~WCnt) && (W[0] || !WCnt[0]);

                if(WLst) WPhase <= '{ act: CROP_W == 0, post: 0 };
                else if(!WPhase.post) begin
                    unique case(WPhase.act)
                        0: if(!((CROP_W - 1) & ~WCnt))
                            WPhase <= '{ act: 1, post: 0 };
                        1: if(!((W-CROP_E-1) & ~WCnt))
                            WPhase <= '{ act: 0, post: 1 };
                    endcase
                end
            end
        end
        assign wlst = WLst;
        assign wact = WPhase.act;
    end : genWCnt

    // Height
    uwire hlst;
    uwire hact;
    if(H < 2) begin : genHTrivial
        assign hlst = 1;
        assign hact = 1;
    end : genHTrivial
    else begin : genHCnt
        //                 | <------ active ------> |
        // 0, ..., CROP_N-1, CROP_N, ..., H-CROP_S-1, H-CROP_S, ..., H-1 (last)
        logic [$clog2(H)-1:0] HCnt = 0;
        logic HLst = 0;
        phase_t HPhase = '{ act: CROP_N == 0, post: 0 };
        always_ff @(posedge clk) begin
            if(rst) begin
                HCnt <= 0;
                HLst <= 0;
                HPhase <= '{ act: CROP_N == 0, post: 0 };
            end
            else if(take && cf_last && wlst) begin
                HCnt <= HCnt + (HLst? -H+1 : 1);
                HLst <= !((H - 2) & ~HCnt) && (H[0] || !HCnt[0]);

                if(HLst) HPhase <= '{ act: CROP_N == 0, post: 0 };
                else if(!HPhase.post) begin
                    unique case(HPhase.act)
                        0: if(!((CROP_N - 1) & ~HCnt))
                            HPhase <= '{ act: 1, post: 0 };
                        1: if(!((H-CROP_S-1) & ~HCnt))
                            HPhase <= '{ act: 0, post: 1 };
                    endcase
                end
            end
        end
        assign hlst = HLst;
        assign hact = HPhase.act;
    end : genHCnt

    uwire selected = hact && wact;

    //=== Selected-Fold Forwarding ==========================================
    data_t ADat = 'x;
    logic AVld = 0;
    data_t BDat = 'x;
    logic BVld = 0;
    logic IRdy = 1;

    uwire take = IRdy && ivld;
    uwire selected_take = selected && take;
    uwire bload = !BVld || ordy;

    always_ff @(posedge clk) begin
        if(rst) begin
            ADat <= 'x;
            AVld <= 0;
            BDat <= 'x;
            BVld <= 0;
            IRdy <= 1;
        end
        else begin
            if(!AVld) ADat <= idat;
            AVld <= !bload && (AVld || selected_take);

            if(bload) begin
                BDat <= AVld? ADat : idat;
                BVld <= AVld || selected_take;
            end

            IRdy <= !selected || bload || (!selected_take && !AVld);
        end
    end
    assign irdy = IRdy;
    assign odat = BDat;
    assign ovld = BVld;

endmodule : crop
