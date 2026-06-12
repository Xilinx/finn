/**
 * Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
 * SPDX-License-Identifier: MIT
 *
 * This file contains the interrupt handler function which must be bound by
 * application software to the interrupt system being used
 * Interrupts from the SMBus IP will trigger a call to this interrupt handler
 * which in turn will drive the state machine
 *
 * @file smbus_interrupt_handler.c
 *
 */

#include "smbus.h"
#include "smbus_internal.h"
#include "smbus_interrupt_handler.h"
#include "smbus_event.h"
#include "smbus_hardware_access.h"

/*******************************************************************************
*
* @brief    Walk through the list of active instances and determine which
*           instance the address corresponds to
*
* @param    pxSMBusProfile is a pointer to the SMBus profile structure.
*
* @return   uint8_t instance number 0-7
*
* @note     None.
*
*******************************************************************************/
static uint8_t ucSMBusDetermineInstanceFromTargetAddress( SMBUS_PROFILE_TYPE* pxSMBusProfile );

/*******************************************************************************
*
* @brief    Function clears any set bits in the ISR and ERR_ISR registers
*
* @param    pxSMBusProfile is a pointer to the SMBus profile structure.
* @param    ulISR_RegisterValue is previous value of the ISR_Register
* @param    ulERR_ISR_RegisterValue is previous value of the ERR_ISR_Register
*
* @return   void
*
* @note     None.
*
*******************************************************************************/
static void prvvSMBusClearInterrupts( SMBUS_PROFILE_TYPE* pxSMBusProfile, uint32_t ulISR_RegisterValue,
                                    uint32_t ulERR_ISR_RegisterValue );

/*******************************************************************************
*
* @brief    Walk through the list of active instances and determine which
*           instance the address corresponds to
*
*******************************************************************************/
static uint8_t ucSMBusDetermineInstanceFromTargetAddress( SMBUS_PROFILE_TYPE* pxSMBusProfile )
{
    uint8_t ucInstance = SMBUS_INVALID_INSTANCE;
    uint8_t ucAddress  = 0;
    int     i          = 0;

    if( NULL != pxSMBusProfile )
    {
        ucAddress = ulSMBusHWReadTgtStatusAddress( pxSMBusProfile );
        for( i = 0; i < SMBUS_NUMBER_OF_SMBUS_INSTANCES; i++ )
        {
            if( SMBUS_TRUE == pxSMBusProfile->xSMBusInstance[i].ucInstanceInUse )
            {
                if( ucAddress == pxSMBusProfile->xSMBusInstance[i].ucSMBusAddress )
                {
                    ucInstance = i;
                    break;
                }
            }
        }
    }

    return ( ucInstance );
}

/*******************************************************************************
*
* @brief    Function clears any set bits in the ISR and ERR_ISR registers
*
*******************************************************************************/
static void prvvSMBusClearInterrupts( SMBUS_PROFILE_TYPE* pxSMBusProfile, uint32_t ulISR_RegisterValue, 
                                    uint32_t ulERR_ISR_RegisterValue )
{
    if( NULL != pxSMBusProfile )
    {
        /* Clear any raised interrupts */
        vSMBusHWWriteERRIRQISR( pxSMBusProfile, ulERR_ISR_RegisterValue );
        vSMBusHWWriteIRQISR( pxSMBusProfile, ulISR_RegisterValue );
    }
}

/*******************************************************************************
*
* @brief    Function will be a callback called from the interrupt handler
*           It will determine what interrupts are present from those add
*           events on the event queue and then trigger the handling of the
*           events by the state machine
*
*******************************************************************************/
void vSMBusInterruptHandler( void* pvCallBackRef )
{
    uint32_t ulISR_RegisterValue        = 0;
    uint32_t ulIER_RegisterValue        = 0;
    uint32_t ulInterruptVector          = 0;
    uint32_t ulERR_ISR_RegisterValue    = 0;
    uint32_t ulERR_IER_RegisterValue    = 0;
    uint32_t ulInterruptErrorVector     = 0;
    uint32_t ulDisableSMBClkLowRegValue = 0;
    uint32_t ulDisableSMBDatLowRegValue = 0;
    uint8_t  ucInstance                 = SMBUS_INVALID_INSTANCE;

    if( NULL != pvCallBackRef )
    {
        SMBUS_PROFILE_TYPE* pxSMBusProfile = ( SMBUS_PROFILE_TYPE* )pvCallBackRef;

        /* check clk/dat status */
        if( ( SMBUS_SMBCLK_LOW_TIMEOUT_DETECTED != ulSMBusHWReadPHYStatusSMBClkLowTimeout( pxSMBusProfile ) ) &&
            ( SMBUS_SMBDAT_LOW_TIMEOUT_DETECTED != ulSMBusHWReadPHYStatusSMBDATLowTimeout( pxSMBusProfile ) ) )
        {
            /* Re-enable all ERR_IRQ_IER interrupts */
            vSMBusHWWriteERRIRQIER( pxSMBusProfile, 0x000FFFFF );   
        }

        /* Disable the interrupt */
        vSMBusHWWriteIRQGIEEnable( pxSMBusProfile, 0 );

        /* Read registers to determine the interrupt source */
        ulISR_RegisterValue = ulSMBusHWReadIRQISR( pxSMBusProfile );
        ulIER_RegisterValue = ulSMBusHWReadIRQIER( pxSMBusProfile );
        ulInterruptVector = ulISR_RegisterValue & ulIER_RegisterValue;

        if( ulInterruptVector & SMBUS_INTERRUPT_ERROR_IRQ )
        {
            ulERR_ISR_RegisterValue = ulSMBusHWReadErrIRQISR( pxSMBusProfile );
            ulERR_IER_RegisterValue = ulSMBusHWReadErrIRQIER( pxSMBusProfile );
            ulInterruptErrorVector = ulERR_ISR_RegisterValue & ulERR_IER_RegisterValue;

            /* Handle CLK/DAT line low interrupts */
            if( ulInterruptErrorVector & SMBUS_ERROR_INTERRUPT_PHY_SMBCLK_LOW_TIMEOUT )
            {
                /* Disable PHY_SMBCLK_LOW_TIMEOUT interrupt */
                ulERR_IER_RegisterValue = ulSMBusHWReadErrIRQIER( pxSMBusProfile );
                ulDisableSMBClkLowRegValue = ulERR_IER_RegisterValue & 0x000FFFFE;
                vSMBusHWWriteERRIRQIER( pxSMBusProfile, ulDisableSMBClkLowRegValue );
            }

            if( ulInterruptErrorVector & SMBUS_ERROR_INTERRUPT_PHY_SMBDAT_LOW_TIMEOUT )
            {
                /* Disable PHY_SMBDAT_LOW_TIMEOUT interrupt */
                ulERR_IER_RegisterValue = ulSMBusHWReadErrIRQIER( pxSMBusProfile );
                ulDisableSMBDatLowRegValue = ulERR_IER_RegisterValue & 0x000FFFFD;
                vSMBusHWWriteERRIRQIER( pxSMBusProfile, ulDisableSMBDatLowRegValue );
            }
        }

        vLogAddEntry( pxSMBusProfile, SMBUS_LOG_LEVEL_INFO, 
                        SMBUS_INSTANCE_UNDETERMINED, SMBUS_LOG_EVENT_INTERRUPT_EVENT, ulISR_RegisterValue, ulERR_ISR_RegisterValue );

        if( 0 != ( ( ulInterruptVector ) & ( SMBUS_INTERRUPT_TGT_INTERRUPTS | SMBUS_INTERRUPT_ERROR_IRQ ) ) )
        {
            if( ( ulInterruptVector ) & ( SMBUS_INTERRUPT_TGT_READ_OR_WRITE_INTERRUPTS ) )
            {
                pxSMBusProfile->ucActiveTargetInstance = ucSMBusDetermineInstanceFromTargetAddress( pxSMBusProfile );
            }

            ucInstance = pxSMBusProfile->ucActiveTargetInstance;
            if( SMBUS_LAST_SMBUS_INSTANCE >= ucInstance )
            {
                if( 0 != ( ( ulInterruptErrorVector ) & ( SMBUS_INTERRUPT_TGT_ERROR_INTERRUPTS ) ) )
                {
                    if( ulInterruptErrorVector & SMBUS_ERROR_INTERRUPT_PHY_TGT_TEXT_TIMEOUT )
                    {
                        vSMBusGenerateEvent_E_TARGET_PHY_TEXT_TIMEOUT_ERROR_IRQ( &( pxSMBusProfile->xSMBusInstance[ucInstance] ) );
                    }

                    if( ulInterruptErrorVector & SMBUS_ERROR_INTERRUPT_TGT_RX_FIFO_ERROR )
                    {
                        vSMBusGenerateEvent_E_TARGET_RX_FIFO_ERROR_ERROR_IRQ( &( pxSMBusProfile->xSMBusInstance[ucInstance] ) );
                    }

                    if( ulInterruptErrorVector & SMBUS_ERROR_INTERRUPT_TGT_RX_FIFO_OVERFLOW )
                    {
                        vSMBusGenerateEvent_E_TARGET_RX_FIFO_OVERFLOW_ERROR_IRQ( &( pxSMBusProfile->xSMBusInstance[ucInstance] ) );
                    }

                    if( ulInterruptErrorVector & SMBUS_ERROR_INTERRUPT_TGT_RX_FIFO_UNDERFLOW )
                    {
                        vSMBusGenerateEvent_E_TARGET_RX_FIFO_UNDERFLOW_ERROR_IRQ( &( pxSMBusProfile->xSMBusInstance[ucInstance] ) );
                    }

                    if( ulInterruptErrorVector & SMBUS_ERROR_INTERRUPT_TGT_DESC_FIFO_ERROR )
                    {
                        vSMBusGenerateEvent_E_TARGET_DESC_FIFO_ERROR_IRQ( &( pxSMBusProfile->xSMBusInstance[ucInstance] ) );
                    }

                    if( ulInterruptErrorVector & SMBUS_ERROR_INTERRUPT_TGT_DESC_FIFO_OVERFLOW )
                    {
                        vSMBusGenerateEvent_E_TARGET_DESC_FIFO_OVERFLOW_ERROR_IRQ( &( pxSMBusProfile->xSMBusInstance[ucInstance] ) );
                    }

                    if( ulInterruptErrorVector & SMBUS_ERROR_INTERRUPT_TGT_DESC_FIFO_UNDERFLOW )
                    {
                        vSMBusGenerateEvent_E_TARGET_DESC_FIFO_UNDERFLOW_ERROR_IRQ( &( pxSMBusProfile->xSMBusInstance[ucInstance] ) );
                    }

                    if( ulInterruptErrorVector & SMBUS_ERROR_INTERRUPT_TGT_DESC_ERROR )
                    {
                        vSMBusGenerateEvent_E_TARGET_DESC_ERROR_IRQ( &( pxSMBusProfile->xSMBusInstance[ucInstance] ) );
                    }

                    if( ulInterruptErrorVector & SMBUS_ERROR_INTERRUPT_PHY_UNEXPTD_BUS_IDLE )
                    {
                        vSMBusGenerateEvent_E_TARGET_PHY_UNEXPTD_BUS_IDLE_ERROR_IRQ( &( pxSMBusProfile->xSMBusInstance[ucInstance] ) );
                    }

                    if( ulInterruptErrorVector & SMBUS_ERROR_INTERRUPT_PHY_SMBDAT_LOW_TIMEOUT )
                    {
                        vSMBusGenerateEvent_E_TARGET_PHY_SMBDAT_LOW_TIMEOUT_DESC_ERROR_IRQ( &( pxSMBusProfile->xSMBusInstance[ucInstance] ) );
                    }

                    if( ulInterruptErrorVector & SMBUS_ERROR_INTERRUPT_PHY_SMBCLK_LOW_TIMEOUT )
                    {
                        vSMBusGenerateEvent_E_TARGET_PHY_SMBCLK_LOW_TIMEOUT_ERROR_IRQ( &( pxSMBusProfile->xSMBusInstance[ucInstance] ) );
                    }
                }

                if( ulInterruptVector & SMBUS_INTERRUPT_TGT_LOA )
                {
                    vSMBusGenerateEvent_E_TARGET_LOA_ERROR_IRQ( &( pxSMBusProfile->xSMBusInstance[ucInstance] ) );
                }

                if( ulInterruptVector & SMBUS_INTERRUPT_TGT_PEC_ERROR )
                {
                    vSMBusGenerateEvent_E_TARGET_PEC_ERROR_IRQ( &( pxSMBusProfile->xSMBusInstance[ucInstance] ) );
                }

                if( ulInterruptVector & SMBUS_INTERRUPT_TGT_READ )
                {
                    vSMBusGenerateEvent_E_TARGET_READ_IRQ( &( pxSMBusProfile->xSMBusInstance[ucInstance] ) );
                }

                if( ulInterruptVector & SMBUS_INTERRUPT_TGT_WRITE )
                {
                    vSMBusGenerateEvent_E_TARGET_WRITE_IRQ( &( pxSMBusProfile->xSMBusInstance[ucInstance] ) );
                }

                if( ulInterruptVector & SMBUS_INTERRUPT_TGT_DESC_FIFO_EMPTY )
                {
                }

                if( ulInterruptVector & SMBUS_INTERRUPT_TGT_RX_FIFO_FILL_THRESHOLD )
                {
                    vSMBusGenerateEvent_E_TARGET_DATA_IRQ( &( pxSMBusProfile->xSMBusInstance[ucInstance] ) );
                }

                if( ulInterruptVector & SMBUS_INTERRUPT_TGT_DONE )
                {
                    vSMBusGenerateEvent_E_TARGET_DONE_IRQ( &( pxSMBusProfile->xSMBusInstance[ucInstance] ) );
                }

                if( ulInterruptVector & SMBUS_INTERRUPT_TGT_DESC_FIFO_ALMOST_EMPTY )
                {
                    vSMBusGenerateEvent_E_DESC_FIFO_ALMOST_EMPTY_IRQ( &( pxSMBusProfile->xSMBusInstance[ucInstance] ) );
                }

                /* Call the event handler for the target */
                vSMBusEventQueueHandle( pxSMBusProfile );
            }
            else
            {
                /* No target instance is currently active - reset */
                vSMBusHWWriteTgtRxFifoReset( pxSMBusProfile, 1 );
            }
        }

        if( 0 != ( ( ulInterruptVector ) & ( SMBUS_INTERRUPT_CTLR_INTERRUPTS | SMBUS_INTERRUPT_ERROR_IRQ ) ) )
        {
            ucInstance = pxSMBusProfile->ucInstanceInPlay;
            if( SMBUS_LAST_SMBUS_INSTANCE >= ucInstance )
            {
                /* Create event and add it to the event queue */
                if( 0 != ( ( ulInterruptErrorVector ) & ( SMBUS_INTERRUPT_CTLR_ERROR_INTERRUPTS ) ) )
                {
                    if( ulInterruptErrorVector & SMBUS_ERROR_INTERRUPT_PHY_CTLR_TEXT_TIMEOUT )
                    {
                        vSMBusGenerateEvent_E_CONTROLLER_PHY_CTLR_TEXT_TIMEOUT_ERROR_IRQ( &( pxSMBusProfile->xSMBusInstance[ucInstance] ) );
                    }

                    if( ulInterruptErrorVector & SMBUS_ERROR_INTERRUPT_PHY_CTLR_CEXT_TIMEOUT )
                    {
                        vSMBusGenerateEvent_E_CONTROLLER_PHY_CTLR_CEXT_TIMEOUT_ERROR_IRQ( &( pxSMBusProfile->xSMBusInstance[ucInstance] ) );
                    }

                    if( ulInterruptErrorVector & SMBUS_ERROR_INTERRUPT_CTLR_RX_FIFO_ERROR )
                    {
                        vSMBusGenerateEvent_E_CONTROLLER_RX_FIFO_ERROR_IRQ( &( pxSMBusProfile->xSMBusInstance[ucInstance] ) );
                    }

                    if( ulInterruptErrorVector & SMBUS_ERROR_INTERRUPT_CTLR_RX_FIFO_OVERFLOW )
                    {
                        vSMBusGenerateEvent_E_CONTROLLER_RX_FIFO_OVERFLOW_ERROR_IRQ( &( pxSMBusProfile->xSMBusInstance[ucInstance] ) );
                    }

                    if( ulInterruptErrorVector & SMBUS_ERROR_INTERRUPT_CTLR_RX_FIFO_UNDERFLOW )
                    {
                        vSMBusGenerateEvent_E_CONTROLLER_RX_FIFO_UNDERFLOW_ERROR_IRQ( &( pxSMBusProfile->xSMBusInstance[ucInstance] ) );
                    }

                    if( ulInterruptErrorVector & SMBUS_ERROR_INTERRUPT_CTLR_DESC_FIFO_ERROR )
                    {
                        vSMBusGenerateEvent_E_CONTROLLER_DESC_FIFO_ERROR_IRQ( &( pxSMBusProfile->xSMBusInstance[ucInstance] ) );
                    }

                    if( ulInterruptErrorVector & SMBUS_ERROR_INTERRUPT_CTLR_DESC_FIFO_OVERFLOW )
                    {
                        vSMBusGenerateEvent_E_CONTROLLER_DESC_FIFO_OVERFLOW_ERROR_IRQ( &( pxSMBusProfile->xSMBusInstance[ucInstance] ) );
                    }

                    if( ulInterruptErrorVector & SMBUS_ERROR_INTERRUPT_CTLR_DESC_FIFO_UNDERFLOW )
                    {
                        vSMBusGenerateEvent_E_CONTROLLER_DESC_FIFO_UNDERFLOW_ERROR_IRQ( &( pxSMBusProfile->xSMBusInstance[ucInstance] ) );
                    }

                    if( ulInterruptErrorVector & SMBUS_ERROR_INTERRUPT_CTLR_DESC_ERROR )
                    {
                        vSMBusGenerateEvent_E_CONTROLLER_DESC_ERROR_IRQ( &( pxSMBusProfile->xSMBusInstance[ucInstance] ) );
                    }
                }

                if( ulInterruptVector & SMBUS_INTERRUPT_CTLR_LOA )
                {
                    vSMBusGenerateEvent_E_CONTROLLER_LOA_ERROR_IRQ( &( pxSMBusProfile->xSMBusInstance[ucInstance] ) );
                }

                if( ulInterruptVector & SMBUS_INTERRUPT_CTLR_NACK_ERROR )
                {
                    vSMBusGenerateEvent_E_CONTROLLER_NACK_ERROR_IRQ( &( pxSMBusProfile->xSMBusInstance[ucInstance] ) );
                }

                if( ulInterruptVector & SMBUS_INTERRUPT_CTLR_PEC_ERROR )
                {
                    vSMBusGenerateEvent_E_CONTROLLER_PEC_ERROR_IRQ( &( pxSMBusProfile->xSMBusInstance[ucInstance] ) );
                }

                if( ulInterruptVector & SMBUS_INTERRUPT_CTLR_DESC_FIFO_EMPTY )
                {
                }

                if( ulInterruptVector & SMBUS_INTERRUPT_CTLR_RX_FIFO_FILL_THRESHOLD )
                {
                    vSMBusGenerateEvent_E_CONTROLLER_DATA_IRQ( &( pxSMBusProfile->xSMBusInstance[ucInstance] ) );
                }

                if( ulInterruptVector & SMBUS_INTERRUPT_CTLR_DESC_FIFO_ALMOST_EMPTY )
                {
                    vSMBusGenerateEvent_E_CONTROLLER_DESC_FIFO_ALMOST_EMPTY_IRQ( &( pxSMBusProfile->xSMBusInstance[ucInstance] ) );
                }

                /* Moved this to last to prevent done happening before receive data */
                if( ulInterruptVector & SMBUS_INTERRUPT_CTLR_DONE )
                {
                    vSMBusGenerateEvent_E_CONTROLLER_DONE_IRQ( &( pxSMBusProfile->xSMBusInstance[ucInstance] ) );
                }

                /* Call the event handler for the controller */
                vSMBusEventQueueHandle( pxSMBusProfile );
            }
            else
            {
                /* No controller instance is currently active - reset */
                vSMBusHWWriteCtrlRxFifoReset( pxSMBusProfile, 1 );
                vSMBusHWWriteCtrlDescFifoReset( pxSMBusProfile, 1 );
            }
        }

        prvvSMBusClearInterrupts( pxSMBusProfile, ulISR_RegisterValue, ulERR_ISR_RegisterValue );

        /* For debug disable all the Controller interrupts as soon as I receive CTLR_DONE */
        if( SMBUS_INVALID_INSTANCE != ucInstance )
        {
            if( ulInterruptVector & SMBUS_INTERRUPT_CTLR_DONE )
            {
                vSMBusHWWriteIRQIER( pxSMBusProfile, 0x000001EF );
            }
        }

        /* Re-enable the interrupt */
        vSMBusHWWriteIRQGIEEnable( pxSMBusProfile, 1 );
    }
}
