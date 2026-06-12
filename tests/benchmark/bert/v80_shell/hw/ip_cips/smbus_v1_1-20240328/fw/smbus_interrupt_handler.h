/**
 * Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
 * SPDX-License-Identifier: MIT
 *
 * This file contains the defintions of the SMBus Interrupts
 *
 * @file smbus_interrupt_handler.h
 *
 */

#ifndef _SMBUS_INTERRUPT_HANDLER_H_
#define _SMBUS_INTERRUPT_HANDLER_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include "smbus.h"

#define SMBUS_INTERRUPT_CTLR_DESC_FIFO_ALMOST_EMPTY     ( 1 << 15 )
#define SMBUS_INTERRUPT_CTLR_RX_FIFO_FILL_THRESHOLD     ( 1 << 14 )
#define SMBUS_INTERRUPT_CTLR_DESC_FIFO_EMPTY            ( 1 << 13 )
#define SMBUS_INTERRUPT_CTLR_DONE                       ( 1 << 12 )
#define SMBUS_INTERRUPT_CTLR_PEC_ERROR                  ( 1 << 11 )
#define SMBUS_INTERRUPT_CTLR_NACK_ERROR                 ( 1 << 10 )
#define SMBUS_INTERRUPT_CTLR_LOA                        ( 1 << 9  )
#define SMBUS_INTERRUPT_TGT_DESC_FIFO_ALMOST_EMPTY      ( 1 << 8  )
#define SMBUS_INTERRUPT_TGT_WRITE                       ( 1 << 7  )
#define SMBUS_INTERRUPT_TGT_READ                        ( 1 << 6  )
#define SMBUS_INTERRUPT_TGT_RX_FIFO_FILL_THRESHOLD      ( 1 << 5  )
#define SMBUS_INTERRUPT_TGT_DESC_FIFO_EMPTY             ( 1 << 4  )
#define SMBUS_INTERRUPT_TGT_DONE                        ( 1 << 3  )
#define SMBUS_INTERRUPT_TGT_PEC_ERROR                   ( 1 << 2  )
#define SMBUS_INTERRUPT_TGT_LOA                         ( 1 << 1  )
#define SMBUS_INTERRUPT_ERROR_IRQ                       ( 1 << 0  )
#define SMBUS_ERROR_INTERRUPT_PHY_CTLR_CEXT_TIMEOUT     ( 1 << 19 )
#define SMBUS_ERROR_INTERRUPT_PHY_CTLR_TEXT_TIMEOUT     ( 1 << 18 )
#define SMBUS_ERROR_INTERRUPT_CTLR_RX_FIFO_ERROR        ( 1 << 17 )
#define SMBUS_ERROR_INTERRUPT_CTLR_RX_FIFO_OVERFLOW     ( 1 << 16 )
#define SMBUS_ERROR_INTERRUPT_CTLR_RX_FIFO_UNDERFLOW    ( 1 << 15 )
#define SMBUS_ERROR_INTERRUPT_CTLR_DESC_FIFO_ERROR      ( 1 << 14 )
#define SMBUS_ERROR_INTERRUPT_CTLR_DESC_FIFO_OVERFLOW   ( 1 << 13 )
#define SMBUS_ERROR_INTERRUPT_CTLR_DESC_FIFO_UNDERFLOW  ( 1 << 12 )
#define SMBUS_ERROR_INTERRUPT_CTLR_DESC_ERROR           ( 1 << 11 )
#define SMBUS_ERROR_INTERRUPT_PHY_TGT_TEXT_TIMEOUT      ( 1 << 10 )
#define SMBUS_ERROR_INTERRUPT_TGT_RX_FIFO_ERROR         ( 1 << 9  )
#define SMBUS_ERROR_INTERRUPT_TGT_RX_FIFO_OVERFLOW      ( 1 << 8  )
#define SMBUS_ERROR_INTERRUPT_TGT_RX_FIFO_UNDERFLOW     ( 1 << 7  )
#define SMBUS_ERROR_INTERRUPT_TGT_DESC_FIFO_ERROR       ( 1 << 6  )
#define SMBUS_ERROR_INTERRUPT_TGT_DESC_FIFO_OVERFLOW    ( 1 << 5  )
#define SMBUS_ERROR_INTERRUPT_TGT_DESC_FIFO_UNDERFLOW   ( 1 << 4  )
#define SMBUS_ERROR_INTERRUPT_TGT_DESC_ERROR            ( 1 << 3  )
#define SMBUS_ERROR_INTERRUPT_PHY_UNEXPTD_BUS_IDLE      ( 1 << 2  )
#define SMBUS_ERROR_INTERRUPT_PHY_SMBDAT_LOW_TIMEOUT    ( 1 << 1  )
#define SMBUS_ERROR_INTERRUPT_PHY_SMBCLK_LOW_TIMEOUT    ( 1 << 0  )

#define SMBUS_INTERRUPT_TGT_INTERRUPTS                  SMBUS_INTERRUPT_TGT_DESC_FIFO_ALMOST_EMPTY      |\
                                                        SMBUS_INTERRUPT_TGT_WRITE                       |\
                                                        SMBUS_INTERRUPT_TGT_READ                        |\
                                                        SMBUS_INTERRUPT_TGT_RX_FIFO_FILL_THRESHOLD      |\
                                                        SMBUS_INTERRUPT_TGT_DESC_FIFO_EMPTY             |\
                                                        SMBUS_INTERRUPT_TGT_DONE                        |\
                                                        SMBUS_INTERRUPT_TGT_PEC_ERROR                   |\
                                                        SMBUS_INTERRUPT_TGT_LOA                   

#define SMBUS_INTERRUPT_CTLR_INTERRUPTS                 SMBUS_INTERRUPT_CTLR_DESC_FIFO_ALMOST_EMPTY     |\
                                                        SMBUS_INTERRUPT_CTLR_RX_FIFO_FILL_THRESHOLD     |\
                                                        SMBUS_INTERRUPT_CTLR_DESC_FIFO_EMPTY            |\
                                                        SMBUS_INTERRUPT_CTLR_DONE                       |\
                                                        SMBUS_INTERRUPT_CTLR_PEC_ERROR                  |\
                                                        SMBUS_INTERRUPT_CTLR_NACK_ERROR                 |\
                                                        SMBUS_INTERRUPT_CTLR_LOA                   

#define SMBUS_INTERRUPT_CTLR_ERROR_INTERRUPTS           SMBUS_ERROR_INTERRUPT_PHY_CTLR_CEXT_TIMEOUT     |\
                                                        SMBUS_ERROR_INTERRUPT_PHY_CTLR_TEXT_TIMEOUT     |\
                                                        SMBUS_ERROR_INTERRUPT_CTLR_RX_FIFO_ERROR        |\
                                                        SMBUS_ERROR_INTERRUPT_CTLR_RX_FIFO_OVERFLOW     |\
                                                        SMBUS_ERROR_INTERRUPT_CTLR_RX_FIFO_UNDERFLOW    |\
                                                        SMBUS_ERROR_INTERRUPT_CTLR_DESC_FIFO_ERROR      |\
                                                        SMBUS_ERROR_INTERRUPT_CTLR_DESC_FIFO_OVERFLOW   |\
                                                        SMBUS_ERROR_INTERRUPT_CTLR_DESC_FIFO_UNDERFLOW  |\
                                                        SMBUS_ERROR_INTERRUPT_CTLR_DESC_ERROR         

#define SMBUS_INTERRUPT_TGT_ERROR_INTERRUPTS            SMBUS_ERROR_INTERRUPT_PHY_TGT_TEXT_TIMEOUT      |\
                                                        SMBUS_ERROR_INTERRUPT_TGT_RX_FIFO_ERROR         |\
                                                        SMBUS_ERROR_INTERRUPT_TGT_RX_FIFO_OVERFLOW      |\
                                                        SMBUS_ERROR_INTERRUPT_TGT_RX_FIFO_UNDERFLOW     |\
                                                        SMBUS_ERROR_INTERRUPT_TGT_DESC_FIFO_ERROR       |\
                                                        SMBUS_ERROR_INTERRUPT_TGT_DESC_FIFO_OVERFLOW    |\
                                                        SMBUS_ERROR_INTERRUPT_TGT_DESC_FIFO_UNDERFLOW   |\
                                                        SMBUS_ERROR_INTERRUPT_TGT_DESC_ERROR            |\
                                                        SMBUS_ERROR_INTERRUPT_PHY_UNEXPTD_BUS_IDLE      |\
                                                        SMBUS_ERROR_INTERRUPT_PHY_SMBDAT_LOW_TIMEOUT    |\
                                                        SMBUS_ERROR_INTERRUPT_PHY_SMBCLK_LOW_TIMEOUT 

#define SMBUS_INTERRUPT_TGT_READ_OR_WRITE_INTERRUPTS    SMBUS_INTERRUPT_TGT_WRITE                       |\
                                                        SMBUS_INTERRUPT_TGT_READ

#ifdef __cplusplus
}
#endif

#endif /* _SMBUS_INTERRUPT_HANDLER_H_ */
